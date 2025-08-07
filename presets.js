let presetXOffset = 32;
let presetThickness = 2;

const shapes = Object.freeze({
  circular: (y, z) => y * y + z * z,
  linear: (y, z) => y * y,
  square: (y, z) => Math.max(y * y, z * z)
});

const presetSettings = {
  DoubleSlit: { slitWidth: 8, slitSpacing: 32, slitHeight: 64 },
  Aperture: { shape: shapes.circular, radius: commonInitValues.radius, invert: false },
  ZonePlate: { shape: shapes.circular, f: 192, nCutouts: 4 },
  Lens: { thickness: 16, radius: commonInitValues.radius, refractiveIndex: commonInitValues.refractiveIndex, half: 0, outerBarrier: true },
  Vortex: { radius: commonInitValues.radius, refractiveIndex: commonInitValues.refractiveIndex, n: 1, autoThickness: true },
  PowerLens: { radius: commonInitValues.radius, refractiveIndex: commonInitValues.refractiveIndex, n: 2, autoThickness: false, invert: false },
  CircularLens: { radius: commonInitValues.radius, refractiveIndex: commonInitValues.refractiveIndex, autoThickness: false, invert: false },
}

/**
 * Updates the wave speed field at a given coordinate
 * @param {Number} x X coordinate
 * @param {Number} yRel Y coordinate relative to y midpoint
 * @param {Number} zRel Z coordinate relative to z midpoint
 * @param {Number} newSpeed New wave speed at the specified coordinate
 */
function updateQuadSymmetry(x, yRel, zRel, newSpeed) {
  if (x < simulationDomain[0] && x >= 0
    && Math.abs(yRel) < yMidpt
    && Math.abs(zRel) < zMidpt
  )
    [
      index3d(x, yMidpt - yRel, zMidpt - zRel),
      index3d(x, yMidpt - yRel, zMidpt + zRel),
      index3d(x, yMidpt + yRel, zMidpt - zRel),
      index3d(x, yMidpt + yRel, zMidpt + zRel)
    ].forEach(i => waveSpeedData[i] = newSpeed);
}

/**
 * Writes the speed texture to the gpu
 * @param {Boolean} reset Whether to reset the wave speed field to 1 
 */
function updateSpeedTexture(reset = false) {
  cleared = reset;
  device.queue.writeTexture(
    { texture: textures.speedTex },
    reset ? waveSpeedData.fill(1) : waveSpeedData,
    { offset: 0, bytesPerRow: simulationDomain[0] * 4, rowsPerImage: simulationDomain[1] },
    { width: simulationDomain[0], height: simulationDomain[1], depthOrArrayLayers: simulationDomain[2] },
  );
}

const lerp = (value, in_min, in_max, out_min, out_max) => ((value - in_min) * (out_max - out_min)) / (in_max - in_min) + out_min;
const mod = (x, a) => x - a * Math.floor(x / a);

const flatPresets = Object.freeze({
  DoubleSlit: (y, z, args = presetSettings.DoubleSlit) => (
    y > args.slitHeight / 2 // fill outside of slit area
      || (y <= args.slitHeight / 2 // fill if inside slit height and outside slit opening
        && (z < (args.slitSpacing - args.slitWidth) / 2 || z > (args.slitSpacing + args.slitWidth) / 2)
      ) ? -1 : 1
  ),
  // cutout grid / 2d version of double slit
  Aperture: (y, z, args = presetSettings.Aperture) => (args.shape(y, z) >= args.radius * args.radius) ? args.invert ? 1 : -1 : args.invert ? -1 : 1,
  ZonePlate: (y, z, args = presetSettings.ZonePlate) => {
    const a = args.shape(y, z);
    const maxN = 2 * args.nCutouts;
    const zone = (n) => n * waveSettings.wavelength * (args.f + n * waveSettings.wavelength / 4);
    for (let n = 1; n <= maxN; n += 2)
      if (a <= zone(n) && a >= zone(n - 1)) return -1;
    return a >= zone(maxN) ? -1 : 1;
  },
});

const phasePlatePresets = Object.freeze({
  Vortex: (y, z, args = presetSettings.Vortex) => { // n is nonzero integer
    const n = Math.PI / args.n;
    return 1 / lerp(mod(Math.atan2(z, y), (2 * n)), 0, 2 * n, 1, args.refractiveIndex);
  },
  PowerLens: (y, z, args = presetSettings.PowerLens) => { // n is positive or 0
    const rNorm = Math.hypot(y, z) / (args.radius);
    // return rNorm ** args.n * (1 - 1 / args.refractiveIndex) + 1 / args.refractiveIndex;
    const a = 1 / ((1 - rNorm ** args.n) * (args.refractiveIndex - 1) + 1);
    return args.invert ? 1 + (1 / args.refractiveIndex) - a : a;
  },
  CircularLens: (y, z, args = presetSettings.CircularLens) => {
    const rNorm = Math.hypot(y, z) / (args.radius);
    const t = 1 + 1 / (2 * args.refractiveIndex * (args.refractiveIndex - 1));
    const a = t - Math.sqrt(t * t - rNorm * rNorm);
    return args.invert ? 1 - a : (1 / args.refractiveIndex + a);
    // return 1 / (Math.sqrt(1 - rNorm*rNorm) * (args.refractiveIndex - 1) + 1);
  }
});

/**
 * Updates the wave speed field to include a 4 way symmetric barrier
 * @param {Function} preset Boolean function -> true: add a barrier, false: set wave speed to default 1
 * @param {Number} distance X distance from x=0
 * @param {Number} thickness Thickness of the barrier
 * @param {Object} args Object containing the arguments for the selected preset
 */
function quadSymmetricFlatPreset(preset, distance = 64, thickness = 2, args) {
  for (let z = 0; z < zMidpt; z++) {
    for (let y = 0; y < yMidpt; y++) {
      for (let x = distance; x < distance + thickness; x++) {
        updateQuadSymmetry(x, y, z, preset(y, z, args));
      }
    }
  }
  updateSpeedTexture();
}

/**
 * Updates the wave speed field to a phase plate
 * @param {Function} preset Function that returns the wave speed value at the given point
 * @param {Number} distance X distance from x=0
 * @param {Object} args Object containing arguments for the selected preset
 */
function phasePlate(preset, distance = 64, args) {
  const thickness = args.autoThickness ? waveSettings.wavelength / (args.refractiveIndex - 1) : gui.io.lensThickness.value; // effective path length difference of 1 wavelength
  for (let z = 0; z < simulationDomain[2]; z++) {
    for (let y = 0; y < simulationDomain[1]; y++) {
      for (let x = 0; x < thickness; x++) {
        waveSpeedData[index3d(x + distance, y, z)] = (shapes.circular(y - yMidpt, z - zMidpt) < args.radius * args.radius)
          ? preset(y - yMidpt, z - zMidpt, args)
          : -1;
      }
    }
  }
  updateSpeedTexture();
}

const lensPresets = Object.freeze({
  elliptical: (x, y, z, radius, thickness, convex) => (y * y + z * z) / (1 - (x * x) / (thickness * thickness)) < radius * radius ? convex : !convex,
  parabolic: (x, y, z, radius, thickness, convex) => x < thickness * (1.1 - (z * z + y * y) / (radius * radius)) ? convex : !convex,
});


/**
 * Updates the wave speed field to a specified lens
 * @param {Function} preset Function -> new wave speed given coordinates
 * @param {Number} distance X distance from x=0
 * @param {Object} args Object containing the arguments for the selected preset
 */
function createLens(preset, convex = true, distance = 64, args = presetSettings.Lens) {
  const halfThickness = Math.ceil(args.thickness / 2);
  const speed = 1 / args.refractiveIndex;
  // run the preset test function for each point in range
  for (let z = 0; z < zMidpt; z++) {
    for (let y = 0; y < yMidpt; y++) {
      for (let x = 0; x < halfThickness; x++) {
        let offset = convex ? 0 : halfThickness + 2;
        let newSpeed = 1;
        let aperture = flatPresets.Aperture(y, z, { shape: shapes.circular, radius: args.radius, invert: false }) < 0;
        if (preset(x, y, z, args.radius, halfThickness, convex) && !aperture) {
          newSpeed = speed;
        } else if (args.outerBarrier && aperture && x < 2) {
          newSpeed = -1;
          offset = 0;
        }
        // write half-lenses
        if (args.half >= 0) updateQuadSymmetry(distance + x - offset, y, z, newSpeed);
        if (args.half <= 0) updateQuadSymmetry(distance - x + offset, y, z, newSpeed);
      }
    }
  }
  updateSpeedTexture();
}