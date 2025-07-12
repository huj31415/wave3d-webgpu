// uniform layout
// 00-15: mat4x4f inv proj*view
// 16-19: vec3f cameraPos, f32 dt
// 20-23: vec3f volSize, f32 rayDelta
// 24-27: vec3f volSizeNorm, f32 padding
// 28-31: vec2f resolution, f32 amp, f32 wavelength
// 32-35: f32 intensityFilter, f32 intensityMultiplier, vec2f padding
// total 36 * f32 = 144 bytes

const uniformStruct = `
    struct Uniforms {
      invMatrix: mat4x4f,   // inverse proj*view matrix
      cameraPos: vec3f,     // camera position in world space
      dt: f32,              // simulation time step
      volSize: vec3f,       // volume size in voxels
      rayDtMult: f32,       // raymarch sampling factor
      volSizeNorm: vec3f,   // normalized volume size (volSize / max(volSize))
      resolution: vec2f,    // canvas resolution: x-width, y-height
      waveSettings: vec2f,  // x: amplitude, y: wavelength
      intensityFilter: f32, // intensity filter strength, 0 = off
      intensityMult: f32,   // intensity rendering multiplier
    };
  `;

const uniformValues = new Float32Array(36);

const kMatrixOffset = 0;
const kCamPosOffset = 16;
const kDtOffset = 19;
const kVolSizeOffset = 20;
const kRayDtMultOffset = 23;
const kVolSizeNormOffset = 24;
const kResOffset = 28;
const kWaveSettingsOffset = 30;
const kIntensityFilterOffset = 32;
const kIntensityMultOffset = 33;

const uni = {};

uni.matrixValue = uniformValues.subarray(kMatrixOffset, kMatrixOffset + 16);
uni.cameraPosValue = uniformValues.subarray(kCamPosOffset, kCamPosOffset + 3);
uni.dtValue = uniformValues.subarray(kDtOffset, kDtOffset + 1);
uni.volSizeValue = uniformValues.subarray(kVolSizeOffset, kVolSizeOffset + 3);
uni.rayDtMultValue = uniformValues.subarray(kRayDtMultOffset, kRayDtMultOffset + 1);
uni.volSizeNormValue = uniformValues.subarray(kVolSizeNormOffset, kVolSizeNormOffset + 3);
uni.resValue = uniformValues.subarray(kResOffset, kResOffset + 2);
uni.waveSettingsValue = uniformValues.subarray(kWaveSettingsOffset, kWaveSettingsOffset + 2);
uni.intensityFilterValue = uniformValues.subarray(kIntensityFilterOffset, kIntensityFilterOffset + 1);
uni.intensityMultValue = uniformValues.subarray(kIntensityMultOffset, kIntensityMultOffset + 1);

Object.freeze(uni);

let dt = 0.5;
let oldDt;

let defaultIntensityFilterStrength = 50;
let intensityFilterStrength = defaultIntensityFilterStrength;

let amp = 1, wavelength = 6;

// simulation domain size [x, y, z], ex. [384, 256, 256], [512, 256, 384]
const simulationDomain = [768, 384, 384];
let newDomainSize = vec3.clone(simulationDomain);
let simVoxelCount = simulationDomain[0] * simulationDomain[1] * simulationDomain[2];

let yMidpt = Math.floor(simulationDomain[1] / 2);
let zMidpt = Math.floor(simulationDomain[2] / 2);

const simulationDomainNorm = simulationDomain.map(v => v / Math.max(...simulationDomain));
let waveSpeedData = new Float32Array(simulationDomain[0] * simulationDomain[1] * simulationDomain[2]).fill(1);

/**
 * Resizes the simulation domain
 * @param {Array<Number>} newSize New simulation domain size
 */
function resizeDomain(newSize) {
  vec3.clone(newSize, simulationDomain);
  vec3.clone(simulationDomain.map(v => v / Math.max(...simulationDomain)), simulationDomainNorm);
  simVoxelCount = simulationDomain[0] * simulationDomain[1] * simulationDomain[2]
  waveSpeedData = new Float32Array(simVoxelCount).fill(1);
  yMidpt = Math.floor(simulationDomain[1] / 2);
  zMidpt = Math.floor(simulationDomain[2] / 2);
  camera.target = vec3.scale(simulationDomainNorm, 0.5);
}

/**
 * Refreshes the active preset
 */
function refreshPreset() {
  const presetType = gui.io.presetSelect.value;
  switch (presetType) {
    case "DoubleSlit":
    case "Aperture":
    case "ZonePlate":
      symmetricFlatBarrier(flatPresets[presetType], presetXOffset, presetThickness, presetSettings[presetType]);
      break;
    case "Lens":
      createLens(lensPresets[gui.io.lensType()], presetXOffset, presetSettings.Lens);
      break;
  }
}

let timeBuffer;

const canvas = document.getElementById("canvas");

const gui = new GUI("3D wave sim on WebGPU", canvas);

// Performance section
gui.addGroup("perf", "Performance");
gui.addStringOutput("res", "Resolution", "", "perf");
gui.addHalfWidthGroups("perfL", "perfR", "perf");
gui.addNumericOutput("fps", "FPS", "", 1, "perfL");
gui.addNumericOutput("frameTime", "Frame", "ms", 2, "perfL");
gui.addNumericOutput("jsTime", "JS", "ms", 2, "perfL");
gui.addNumericOutput("computeTime", "Compute", "ms", 2, "perfR");
gui.addNumericOutput("renderTime", "Render", "ms", 2, "perfR");

// Camera state section
gui.addGroup("camState", "Camera state");
gui.addNumericOutput("camFOV", "FOV", "°", 2, "camState");
gui.addNumericOutput("camDist", "Dst", "", 2, "camState");
gui.addStringOutput("camTarget", "Tgt", "", "camState");
gui.addStringOutput("camPos", "Pos", "", "camState");
gui.addNDimensionalOutput(["camAlt", "camAz"], "Alt/az", "°", ", ", 2, "camState");

// Sim controls
gui.addGroup("simCtrl", "Sim controls");
gui.addNumericInput("dt", true, "dt", 0, 1, 0.01, 0.5, 2, "simCtrl", (newDt) => {
  if (oldDt) oldDt = newDt;
  else dt = newDt;
});
gui.addNumericInput("xSize", false, "X size (restart)", 8, 1024, 8, simulationDomain[0], 0, "simCtrl", (value) => newDomainSize[0] = value);
gui.addNumericInput("ySize", false, "Y size (restart)", 8, 512, 8, simulationDomain[1], 0, "simCtrl", (value) => newDomainSize[1] = value);
gui.addNumericInput("zSize", false, "Z size (restart)", 8, 512, 8, simulationDomain[2], 0, "simCtrl", (value) => newDomainSize[2] = value);
gui.addNumericInput("wavelength", true, "Wavelength", 4, 100, 0.1, 6, 1, "simCtrl", (value) => { wavelength = value; uni.waveSettingsValue.set([amp, wavelength]); });
gui.addNumericInput("amp", true, "Amplitude", 0.1, 5, 0.1, 1, 1, "simCtrl", (value) => { amp = value; uni.waveSettingsValue.set([amp, wavelength]); });
gui.addButton("toggleSim", "Play / Pause", false, "simCtrl", () => {
  if (oldDt) {
    dt = oldDt;
    oldDt = null;
  } else {
    oldDt = dt;
    dt = 0;
  }
});

// stops interpolating after restarting?
gui.addButton("restartSim", "Restart", false, "simCtrl", () => {
  cancelAnimationFrame(rafId);
  clearInterval(perfIntId);
  resizeDomain(newDomainSize);
  refreshPreset();
  device.queue.writeBuffer(timeBuffer, 0, new Float32Array([0]));
  main();
});

// Preset controls
gui.addGroup("presets", "Presets");

gui.addRadioOptions("shape", ["circular", "square", "linear"], "circular", "presets", (value) => presetSettings.Aperture.shape = presetSettings.ZonePlate.shape = shapes[value]);
gui.addNumericInput("barrierThickness", true, "Thickness", 1, 16, 1, 2, 0, "presets", (value) => presetThickness = value)

gui.addNumericInput("f", true, "Focal length", 4, 512, 1, 192, 0, "presets", (value) => presetSettings.ZonePlate.f = value);
gui.addNumericInput("nCutouts", true, "# Cutouts", 1, 10, 1, 4, 0, "presets", (value) => presetSettings.ZonePlate.nCutouts = value);

gui.addNumericInput("slitWidth", true, "Slit width", 3, 512, 1, 8, 0, "presets", (value) => presetSettings.DoubleSlit.slitWidth = value);
gui.addNumericInput("slitSpacing", true, "Slit spacing", 0, 512, 1, 64, 0, "presets", (value) => presetSettings.DoubleSlit.slitSpacing = value);
gui.addNumericInput("slitHeight", true, "Slit height", 0, 512, 1, 64, 0, "presets", (value) => presetSettings.DoubleSlit.slitHeight = value);

gui.addNumericInput("radius", true, "Radius", 0, 256, 1, 16, 0, "presets", (value) => presetSettings.Aperture.radius = presetSettings.Lens.radius = value);

gui.addCheckbox("invert", "Invert barrier", false, "presets", (checked) => presetSettings.Aperture.invert = checked);

gui.addRadioOptions("lensType", ["elliptical", "parabolic"], "parabolic", "presets");
gui.addNumericInput("lensThickness", true, "Thickness", 4, 100, 1, 16, 0, "presets", (value) => presetSettings.Lens.thickness = value);
gui.addNumericInput("refractiveIndex", false, "Refractive index", 0.5, 2, 0.01, 1.2, 2, "presets", (value) => presetSettings.Lens.refractiveIndex = value);
gui.addNumericInput("halfLens", true, "Half lens", -1, 1, 1, 0, 0, "presets", (value) => presetSettings.Lens.half = value);
gui.addCheckbox("outerBarrier", "Outer barrier", true, "presets", (checked) => presetSettings.Lens.outerBarrier = checked);

gui.addDropdown("presetSelect", "Select preset", ["ZonePlate", "DoubleSlit", "Aperture", "Lens"], "presets", {
  "ZonePlate": ["shape", "f", "nCutouts"],
  "DoubleSlit": ["slitWidth", "slitSpacing", "slitHeight"],
  "Aperture": ["shape", "radius", "invert"],
  "Lens": ["radius", "lensType", "lensThickness", "refractiveIndex", "halfLens", "outerBarrier"],
});
gui.addNumericInput("xOffset", true, "X Offset", 0, 512, 1, 64, 0, "presets", (value) => presetXOffset = value);
gui.addButton("updatePreset", "Load preset", true, "presets", refreshPreset);

// Visualization controls
gui.addGroup("visCtrl", "Visualization controls");
gui.addNumericInput("rayDtMult", true, "Ray dt mult", 0.1, 5, 0.1, 2, 1, "visCtrl", (value) => uni.rayDtMultValue.set([value]));
gui.addCheckbox("intensity", "Visualize intensity", true, "visCtrl", (checked) => {
  intensityFilterStrength = checked ? defaultIntensityFilterStrength : 0;
  uni.intensityFilterValue.set([intensityFilterStrength]);
});
gui.addNumericInput("intensityMult", true, "Intensity mult", 0.01, 5, 0.01, 1, 2, "visCtrl", (value) => uni.intensityMultValue.set([value]));

// Camera keybinds
gui.addGroup("camKeybinds", "Camera controls",
  `<div>
    Orbit: leftclick / arrows
    <br>
    Pan: rightclick / wasdgv
    <br>
    Zoom: scroll / fc
    <br>
    FOV zoom: ctrl+scroll / ctrl+fc
    <br>
    FOV: alt+scroll / alt+fc
    <br>
    Reset view: middleclick / space
    <br>
    Reset FOV: ctrl+middleclick / ctrl+space
  </div>`
);


// requestAnimationFrame id, fps update id
let rafId, perfIntId;


// timing
let jsTime = 0, lastFrameTime = performance.now(), deltaTime = 10, fps = 0,
  waveComputeTime = 0, boundaryComputeTime = 0, renderTime = 0;

window.onresize = window.onload = () => {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  camera.updateMatrix();
  uni.resValue.set([canvas.width, canvas.height]);
  gui.io.res([window.innerWidth, window.innerHeight]);
};

/**
 * Clamps a number between between specified values
 * @param {Number} min Lower bound to clamp
 * @param {Number} max Upper bound to clamp
 * @returns Original number clamped between min and max
 */
Number.prototype.clamp = function (min, max) { return Math.max(min, Math.min(max, this)) };

/**
 * Converts degrees to radians
 * @returns Degree value in radians
 */
Number.prototype.toRad = function () { return this * Math.PI / 180; }

/**
 * Converts radians to degrees
 * @returns Radian value in degrees
 */
Number.prototype.toDeg = function () { return this / Math.PI * 180; }

/**
 * Generates a random number within a range
 * @param {Number} min Lower bound, inclusive
 * @param {Number} max Upper bound, exclusive
 * @returns Random number between [min, max)
 */
const randRange = (min, max) => Math.random() * (max - min) + min;

/**
 * Generates a random number within a range of 0-max
 * @param {Number} max Upper bound, exclusive
 * @returns Random number between [0, max)
 */
const randMax = (max) => Math.random() * max;

/**
 * 
 * @param {Number} x x coordinate
 * @param {Number} y y coordinate
 * @param {Number} z z coordinate
 * @returns Linear index within simulation domain
 */
const index3d = (x, y, z) => x + simulationDomain[0] * (y + z * simulationDomain[1]);