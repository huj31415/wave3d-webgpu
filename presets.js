let speedTex;

let presetXOffset = 64;
let presetThickness = 2;

const shapes = Object.freeze({
  circular: (y, z) => y * y + z * z,
  linear: (y, z) => y * y,
  square: (y, z) => Math.max(y * y, z * z)
});

const presetSettings = {
  DoubleSlit: { slitWidth: 8, slitSpacing: 64, slitHeight: 64 },
  Aperture: { shape: shapes.circular, radius: 16, invert: false },
  ZonePlate: { shape: shapes.circular, f: 192, nCutouts: 4 },
  Lens: { thickness: 16, radius: 64, refractiveIndex: 1.5, half: 0, outerBarrier: true },
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
  device.queue.writeTexture(
    { texture: speedTex },
    reset ? waveSpeedData.fill(1) : waveSpeedData,
    { offset: 0, bytesPerRow: simulationDomain[0] * 4, rowsPerImage: simulationDomain[1] },
    { width: simulationDomain[0], height: simulationDomain[1], depthOrArrayLayers: simulationDomain[2] },
  );
}

const flatPresets = Object.freeze({
  DoubleSlit: (y, z, args = presetSettings.DoubleSlit) => (
    y > args.slitHeight / 2 // fill outside of slit area
    || (y <= args.slitHeight / 2 // fill if inside slit height and outside slit opening
      && (z < (args.slitSpacing - args.slitWidth) / 2 || z > (args.slitSpacing + args.slitWidth) / 2)
    )
  ),
  Aperture: (y, z, args = presetSettings.Aperture) => (args.shape(y, z) >= args.radius * args.radius) ? !args.invert : args.invert,
  ZonePlate: (y, z, args = presetSettings.ZonePlate) => {
    const a = args.shape(y, z);
    const maxN = 2 * args.nCutouts;
    const zone = (n) => n * wavelength * (args.f + n * wavelength / 4);
    for (let n = 1; n <= maxN; n += 2)
      if (a <= zone(n) && a >= zone(n - 1)) return true;
    return a >= zone(maxN);
  },
});

/**
 * Updates the wave speed field to include a barrier
 * @param {Function} preset Boolean function -> true: add a barrier, false: set wave speed to default 1
 * @param {Number} distance X distance from x=0
 * @param {Number} thickness Thickness of the barrier
 * @param {Object} args Object containing the arguments for the selected preset
 */
function symmetricFlatBarrier(preset, distance = 64, thickness = 2, args) {
  for (let z = 0; z < zMidpt; z++) {
    for (let y = 0; y < yMidpt; y++) {
      for (let x = distance; x < distance + thickness; x++) {
        updateQuadSymmetry(x, y, z, preset(y, z, args) ? -1 : 1);
      }
    }
  }
  updateSpeedTexture();
}

const lensPresets = Object.freeze({
  elliptical: (x, y, z, radius, thickness) => (y * y + z * z) / (1 - (x * x) / (thickness * thickness)) < radius * radius,
  parabolic: (x, y, z, radius, thickness) => x < thickness * (1 - (z * z + y * y) / (radius * radius)),
});


/**
 * Updates the wave speed field to a specified lens
 * @param {Function} preset Function -> new wave speed given coordinates
 * @param {Number} distance X distance from x=0
 * @param {Object} args Object containing the arguments for the selected preset
 */
function createLens(preset, distance = 64, args = presetSettings.Lens) {
  const halfThickness = Math.floor(args.thickness / 2);
  for (let z = 0; z < zMidpt; z++) {
    for (let y = 0; y < yMidpt; y++) {
      for (let x = 0; x < halfThickness; x++) {
        let newSpeed = 1;
        if (preset(x, y, z, args.radius, halfThickness)) {
          newSpeed = 1 / args.refractiveIndex;
        } else if (args.outerBarrier && flatPresets.Aperture(y, z, { shape: shapes.circular, radius: args.radius, invert: false }) && x < 2) {
          newSpeed = -1;
        }
        if (args.half >= 0) updateQuadSymmetry(distance + x, y, z, newSpeed);
        if (args.half <= 0) updateQuadSymmetry(distance - x, y, z, newSpeed);
      }
    }
  }
  updateSpeedTexture();
}