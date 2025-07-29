// uniform layout
// 00-15: mat4x4f inv proj*view
// 16-19: vec3f cameraPos, f32 dt
// 20-23: vec3f volSize, f32 rayDelta
// 24-27: vec3f volSizeNorm, f32 waveOn
// 28-31: vec2f resolution, vec2f waveSettings (f32 amp, f32 wavelength)
// 32-35: f32 intensityFilter, f32 intensityMultiplier, f32 waveSourceType, f32 waveform
// total 36 * f32 = 144 bytes

const uniformStruct = `
    struct Uniforms {
      invMatrix: mat4x4f,   // inverse proj*view matrix
      cameraPos: vec3f,     // camera position in world space
      dt: f32,              // simulation time step
      volSize: vec3f,       // volume size in voxels
      rayDtMult: f32,       // raymarch sampling factor
      volSizeNorm: vec3f,   // normalized volume size (volSize / max(volSize))
      waveOn: f32,          // whether the wave is on or not
      resolution: vec2f,    // canvas resolution: x-width, y-height
      waveSettings: vec2f,  // x: amplitude, y: wavelength
      intensityFilter: f32, // intensity filter strength, 0 = off
      intensityMult: f32,   // intensity rendering multiplier
      waveSourceType: f32,  // source type: 0=plane, 1=point
      waveform: f32,        // waveform: 0=sine, 1=square, 2=triangle, 3=sawtooth
    };
  `;

const uniformValues = new Float32Array(36);

const kMatrixOffset = 0;
const kCamPosOffset = 16;
const kDtOffset = 19;
const kVolSizeOffset = 20;
const kRayDtMultOffset = 23;
const kVolSizeNormOffset = 24;
const kWaveOnOffset = 27;
const kResOffset = 28;
const kWaveSettingsOffset = 30;
const kIntensityFilterOffset = 32;
const kIntensityMultOffset = 33;
const kWaveSourceTypeOffset = 34;
const kWaveformOffset = 35;

const uni = {};

uni.matrixValue = uniformValues.subarray(kMatrixOffset, kMatrixOffset + 16);
uni.cameraPosValue = uniformValues.subarray(kCamPosOffset, kCamPosOffset + 3);
uni.dtValue = uniformValues.subarray(kDtOffset, kDtOffset + 1);
uni.volSizeValue = uniformValues.subarray(kVolSizeOffset, kVolSizeOffset + 3);
uni.rayDtMultValue = uniformValues.subarray(kRayDtMultOffset, kRayDtMultOffset + 1);
uni.volSizeNormValue = uniformValues.subarray(kVolSizeNormOffset, kVolSizeNormOffset + 3);
uni.waveOnValue = uniformValues.subarray(kWaveOnOffset, kWaveOnOffset + 1);
uni.resValue = uniformValues.subarray(kResOffset, kResOffset + 2);
uni.ampValue = uniformValues.subarray(kWaveSettingsOffset, kWaveSettingsOffset + 1);
uni.wavelengthValue = uniformValues.subarray(kWaveSettingsOffset + 1, kWaveSettingsOffset + 2);
uni.intensityFilterValue = uniformValues.subarray(kIntensityFilterOffset, kIntensityFilterOffset + 1);
uni.intensityMultValue = uniformValues.subarray(kIntensityMultOffset, kIntensityMultOffset + 1);
uni.waveSourceTypeValue = uniformValues.subarray(kWaveSourceTypeOffset, kWaveSourceTypeOffset + 1);
uni.waveformValue = uniformValues.subarray(kWaveformOffset, kWaveformOffset + 1);

Object.freeze(uni);

const textures = {
  stateTex0: null,
  stateTex1: null,
  intensityTex: null,
  speedTex: null,
};

let dt = 0.5;
let oldDt;

let dtPerFrame = 1;

let waveOn = true;

let defaultIntensityFilterStrength = 50;
let intensityFilterStrength = defaultIntensityFilterStrength;

let amp = 1, ampVal = amp, wavelength = 6;

// simulation domain size [x, y, z], ex. [384, 256, 256], [512, 256, 384]
const simulationDomain = [384, 256, 256];//[768, 384, 384];
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
  simVoxelCount = simulationDomain[0] * simulationDomain[1] * simulationDomain[2];
  waveSpeedData = new Float32Array(simVoxelCount).fill(1);
  yMidpt = Math.floor(simulationDomain[1] / 2);
  zMidpt = Math.floor(simulationDomain[2] / 2);
  camera.target = defaults.target = vec3.scale(simulationDomainNorm, 0.5);
}

/**
 * Refreshes the active preset
 */
function refreshPreset(clear = false) {
  if (clear) waveSpeedData.fill(1);
  const presetType = gui.io.presetSelect.value;
  switch (presetType) {
    case "DoubleSlit":
    case "Aperture":
    case "ZonePlate":
      quadSymmetricFlatPreset(flatPresets[presetType], presetXOffset, presetThickness, presetSettings[presetType]);
      break;
    case "Lens":
      createLens(lensPresets[gui.io.lensType()], true, presetXOffset, presetSettings.Lens);
      break;
    case "PhasePlate":
      const plateType = gui.io.phasePlateType();
      phasePlate(phasePlatePresets[plateType], presetXOffset, presetSettings[plateType]);
      break;
  }
}

function softReset() {
  const zeros = new Float32Array(simVoxelCount).fill(0);
  device.queue.writeTexture(
    { texture: textures.stateTex0 },
    zeros,
    { offset: 0, bytesPerRow: simulationDomain[0] * 4, rowsPerImage: simulationDomain[1] },
    { width: simulationDomain[0], height: simulationDomain[1], depthOrArrayLayers: simulationDomain[2] },
  );
  device.queue.writeTexture(
    { texture: textures.stateTex1 },
    zeros,
    { offset: 0, bytesPerRow: simulationDomain[0] * 4, rowsPerImage: simulationDomain[1] },
    { width: simulationDomain[0], height: simulationDomain[1], depthOrArrayLayers: simulationDomain[2] },
  );
  device.queue.writeTexture(
    { texture: textures.intensityTex },
    zeros,
    { offset: 0, bytesPerRow: simulationDomain[0] * 4, rowsPerImage: simulationDomain[1] },
    { width: simulationDomain[0], height: simulationDomain[1], depthOrArrayLayers: simulationDomain[2] },
  );
}

function hardReset() {
  cancelAnimationFrame(rafId);
  clearInterval(perfIntId);
  if (!vec3.equals(simulationDomain, newDomainSize)) resizeDomain(newDomainSize);
  textures.speedTex.destroy();
  main().then(refreshPreset);
}

const waveformOptions = Object.freeze({
  sine: 0,
  square: 1,
  triangle: 2,
  sawtooth: 3
});


const canvas = document.getElementById("canvas");

const commonInitValues = {
  radius: 64,
  refractiveIndex: 1.5,
}

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
gui.addNumericInput("dt", true, "dt", 0, 1, 0.01, dt, 2, "simCtrl", (newDt) => {
  if (oldDt) oldDt = newDt;
  else {
    dt = newDt;
    uni.dtValue.set([dt]);
  }
}, "Simulation delta-time; must meet the CFL condition for stability");
gui.addNumericInput("xSize", false, "X size (reinit)", 8, 1024, 8, simulationDomain[0], 0, "simCtrl", (value) => newDomainSize[0] = value, "Requires reinitialization to apply");
gui.addNumericInput("ySize", false, "Y size (reinit)", 8, 512, 8, simulationDomain[1], 0, "simCtrl", (value) => newDomainSize[1] = value, "Requires reinitialization to apply");
gui.addNumericInput("zSize", false, "Z size (reinit)", 8, 512, 8, simulationDomain[2], 0, "simCtrl", (value) => newDomainSize[2] = value, "Requires reinitialization to apply");
gui.addNumericInput("wavelength", true, "Wavelength", 4, 100, 0.1, 6, 1, "simCtrl", (value) => { wavelength = value; uni.wavelengthValue.set([wavelength]); });
gui.addNumericInput("amp", true, "Amplitude", 0.1, 5, 0.1, 1, 1, "simCtrl", (value) => { amp = ampVal = value; uni.ampValue.set([amp]); });
gui.addHalfWidthGroups("waveformOptions", "sourceTypeOptions", "simCtrl");
gui.addRadioOptions("waveform", ["sine", "square", "triangle", "sawtooth"], "sine", "waveformOptions", (value) => uni.waveformValue.set([waveformOptions[value]]));
gui.addRadioOptions("sourceType", ["plane", "point"], "plane", "sourceTypeOptions", (value) => uni.waveSourceTypeValue.set([value === "plane" ? 0 : 1]));
gui.addButton("waveOn", "Toggle wave generator", true, "simCtrl", () => {
  waveOn = !waveOn;
  if (waveOn) {
    uni.waveOnValue.set([1]);
    amp = ampVal;
    uni.ampValue.set([amp]);
  }
});
gui.addButton("toggleSim", "Play / Pause", false, "simCtrl", () => {
  if (oldDt) {
    dt = oldDt;
    oldDt = null;
    uni.dtValue.set([dt]);
  } else {
    oldDt = dt;
    dt = 0;
  }
});

gui.addButton("softRestart", "Restart", false, "simCtrl", softReset);
gui.addButton("hardRestart", "Reinitialize", true, "simCtrl", hardReset);

// Preset controls
gui.addGroup("presets", "Presets");

gui.addRadioOptions("shape", ["circular", "square", "linear"], "circular", "presets", (value) => presetSettings.Aperture.shape = presetSettings.ZonePlate.shape = shapes[value]);

gui.addNumericInput("f", true, "Focal length", 4, 512, 1, 192, 0, "presets", (value) => presetSettings.ZonePlate.f = value);
gui.addNumericInput("nCutouts", true, "# Cutouts", 1, 20, 1, 4, 0, "presets", (value) => presetSettings.ZonePlate.nCutouts = value);

gui.addNumericInput("slitWidth", true, "Slit width", 3, 512, 1, 8, 0, "presets", (value) => presetSettings.DoubleSlit.slitWidth = value);
gui.addNumericInput("slitSpacing", true, "Slit spacing", 0, 512, 1, 64, 0, "presets", (value) => presetSettings.DoubleSlit.slitSpacing = value);
gui.addNumericInput("slitHeight", true, "Slit height", 0, 512, 1, 64, 0, "presets", (value) => presetSettings.DoubleSlit.slitHeight = value);

gui.addNumericInput("radius", true, "Radius", 0, 256, 1, commonInitValues.radius, 0, "presets", (value) =>
  presetSettings.Aperture.radius = presetSettings.Lens.radius = presetSettings.Vortex.radius = presetSettings.CircularLens.radius = presetSettings.PowerLens.radius = value
);

gui.addCheckbox("invert", "Invert barrier", false, "presets", (checked) => presetSettings.Aperture.invert = checked);

gui.addRadioOptions("lensType", ["elliptical", "parabolic"], "parabolic", "presets");
gui.addNumericInput("lensThickness", true, "Thickness", 4, 100, 1, 16, 0, "presets", (value) =>
  presetSettings.Lens.thickness = presetSettings.CircularLens.thickness = presetSettings.PowerLens.thickness = value
);
gui.addNumericInput("refractiveIndex", false, "Refractive index", 0.5, 2, 0.01, commonInitValues.refractiveIndex, 2, "presets", (value) =>
  presetSettings.Lens.refractiveIndex = presetSettings.Vortex.refractiveIndex = presetSettings.PowerLens.refractiveIndex = presetSettings.CircularLens.refractiveIndex = value
);
gui.addNumericInput("halfLens", true, "Half lens", -1, 1, 1, 0, 0, "presets", (value) => presetSettings.Lens.half = value, "-1: curved toward source, 0: both halves, 1: flat toward source");
gui.addCheckbox("outerBarrier", "Outer barrier", true, "presets", (checked) => presetSettings.Lens.outerBarrier = checked);


gui.addNumericInput("barrierThickness", true, "Thickness", 1, 16, 1, 2, 0, "presets", (value) => presetThickness = value);
gui.addNumericInput("xOffset", true, "X Offset", 0, 512, 1, 16, 0, "presets", (value) => presetXOffset = value);

gui.addGroup("phasePlateOptions-container", null, null, "presets");
gui.addNumericInput("nVortices", true, "n vortices", -4, 4, 1, 1, 0, "phasePlateOptions-container", (value) => presetSettings.Vortex.n = value);
gui.addNumericInput("exp", true, "exp", 0, 5, 0.1, 2, 1, "phasePlateOptions-container", (value) => presetSettings.PowerLens.n = value);
gui.addRadioOptions("phasePlateType", ["Vortex", "PowerLens", "CircularLens"], "Vortex", "phasePlateOptions-container", {
  "Vortex": ["nVortices"],
  "PowerLens": ["exp"],
  "CircularLens": [],
});

gui.addDropdown("presetSelect", "Select preset", ["ZonePlate", "DoubleSlit", "Aperture", "Lens", "PhasePlate"], "presets", {
  "ZonePlate": ["shape", "f", "nCutouts", "barrierThickness"],
  "DoubleSlit": ["slitWidth", "slitSpacing", "slitHeight", "barrierThickness"],
  "Aperture": ["shape", "radius", "invert", "barrierThickness"],
  "Lens": ["radius", "lensType", "lensThickness", "refractiveIndex", "halfLens", "outerBarrier"],
  "PhasePlate": ["radius", "refractiveIndex", "lensThickness", "phasePlateOptions"],
});

gui.addButton("updatePreset", "Load preset", false, "presets", () => refreshPreset(false));
gui.addButton("clearUpdatePreset", "Clear & load", false, "presets", () => refreshPreset(true));
gui.addButton("clearPreset", "Clear", true, "presets", () => updateSpeedTexture(true));

// Visualization controls
gui.addGroup("visCtrl", "Visualization controls");
gui.addNumericInput("rayDtMult", true, "Ray dt mult", 0.1, 5, 0.1, 2, 1, "visCtrl", (value) => uni.rayDtMultValue.set([value]), "Raymarching step multipler; higher has better visual quality, lower has better performance");
gui.addCheckbox("intensity", "Visualize intensity", true, "visCtrl", (checked) => {
  intensityFilterStrength = checked ? defaultIntensityFilterStrength : 0;
  uni.intensityFilterValue.set([intensityFilterStrength]);
});
gui.addNumericInput("intensityMult", true, "Intensity mult", 0.01, 5, 0.01, 1, 2, "visCtrl", (value) => uni.intensityMultValue.set([value]), "Raw intensity value multiplier before transfer function");

// Camera keybinds
gui.addGroup("camKeybinds", "Camera controls", `
  <div>
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
  </div>
`);

// Extra info
gui.addGroup("extraInfo", null, `
  <div>
    Click on section titles to expand/collapse
    <br>
    Hover on numeric input labels for more info if applicable, click to toggle between raw number and slider type input
    <br>
  </div>
`);


// requestAnimationFrame id, fps update id
let rafId, perfIntId;


// timing
let jsTime = 0, lastFrameTime = performance.now(), deltaTime = 10, fps = 0,
  waveComputeTime = 0, boundaryComputeTime = 0, renderTime = 0;

// handle resizing
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