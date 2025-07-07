const symmetricPresets = Object.freeze({
  doubleSlit: (y, z, slitWidth = 8, slitSpacing = 64, slitHeight = 64) => y > slitHeight / 2 || (y <= slitHeight / 2 && (z < (slitSpacing - slitWidth) / 2 || z > (slitSpacing + slitWidth) / 2)),
  circleAperture: (y, z, radius = 16) => y * y + z * z > radius * radius,
  squareAperture: (y, z, radius = 16) => y > radius || z > radius,
});

function quadSymmetricFlatBarrier(presetTest, distance = 64, thickness = 2, args = []) {
  const yMidpt = Math.floor(simulationDomain[1] / 2);
  const zMidpt = Math.floor(simulationDomain[2] / 2);
  for (let z = 0; z < zMidpt; z++) {
    for (let y = 0; y < yMidpt; y++) {
      for (let x = distance; x < distance + thickness; x++) {
        if (presetTest(y, z, ...args)) {
          const index1 = index3d(x, yMidpt - y, zMidpt - z);
          const index2 = index3d(x, yMidpt - y, zMidpt + z);
          const index3 = index3d(x, yMidpt + y, zMidpt - z);
          const index4 = index3d(x, yMidpt + y, zMidpt + z);
          waveSpeedData[index1] = waveSpeedData[index2] = waveSpeedData[index3] = waveSpeedData[index4] = -1;
        }
      }
    }
  }
}