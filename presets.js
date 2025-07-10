let speedTex;

let presetRadius = 16;
let presetXOffset = 64;


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

function updateSpeedTexture(reset = false) {
  device.queue.writeTexture(
    { texture: speedTex },
    reset ? waveSpeedData.fill(1) : waveSpeedData,
    { offset: 0, bytesPerRow: simulationDomain[0] * 4, rowsPerImage: simulationDomain[1] },
    { width: simulationDomain[0], height: simulationDomain[1], depthOrArrayLayers: simulationDomain[2] },
  );
}

const shapes = Object.freeze({
  circular: (y, z) => y * y + z * z,
  linear: (y, z) => y * y,
  square: (y, z) => Math.max(y * y, z * z)
});

const flatPresets = Object.freeze({
  doubleSlit: (y, z, args = { slitWidth: 8, slitSpacing: 64, slitHeight: 64 }) => (
    y > args.slitHeight / 2 // fill outside of slit area
    || (y <= args.slitHeight / 2 // fill if inside slit height and outside slit opening
      && (z < (args.slitSpacing - args.slitWidth) / 2 || z > (args.slitSpacing + args.slitWidth) / 2)
    )
  ),
  aperture: (y, z, args = { shape: shapes.circular, radius: 16 }) => args.shape(y, z) >= args.radius * args.radius,
  zonePlate: (y, z, args = { shape: shapes.circular, f: 32, nCutouts: 10 }) => {
    const a = args.shape(y, z);
    const maxN = 2 * args.nCutouts;
    const zone = (n) => n * wavelength * (args.f + n * wavelength / 4);
    for (let n = 1; n <= maxN; n += 2)
      if (a <= zone(n) && a >= zone(n - 1)) return true;
    return a >= zone(maxN);
  },
});

function symmetricFlatBarrier(presetTest, distance = 64, thickness = 2, args) {
  for (let z = 0; z < zMidpt; z++) {
    for (let y = 0; y < yMidpt; y++) {
      for (let x = distance; x < distance + thickness; x++) {
        if (presetTest(y, z, args)) {
          updateQuadSymmetry(x, y, z, -1);
        }
      }
    }
  }
  updateSpeedTexture();
}

const lensPresets = Object.freeze({
  ellipical: (x, y, z, radius, thickness) => (y * y + z * z) / (1 - (x * x) / (thickness * thickness)) < radius * radius,
  parabolic: (x, y, z, radius, thickness) => x < thickness * (1 - (z * z + y * y) / (radius * radius)),
});


function symmetricLens(presetTest, distance = 64, thickness = 16, radius = 64, refractiveIndex = 1.5, half = 0, outerBarrier = true) {
  const halfThickness = Math.floor(thickness / 2);
  for (let z = 0; z < zMidpt; z++) {
    for (let y = 0; y < yMidpt; y++) {
      for (let x = 0; x < halfThickness; x++) {
        let newSpeed = 1;
        if (presetTest(x, y, z, radius, halfThickness)) {
          newSpeed = 1 / refractiveIndex;
        } else if (outerBarrier && flatPresets.circleAperture(y, z, { radius: radius }) && x < 2) {
          newSpeed = -1;
        }
        if (half >= 0) updateQuadSymmetry(distance + x, y, z, newSpeed);
        if (half <= 0) updateQuadSymmetry(distance - x, y, z, newSpeed);
      }
    }
  }
  updateSpeedTexture();
}