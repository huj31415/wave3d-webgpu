// uniform layout
// 00-15: mat4x4f inv proj*view
// 16-19: vec3f cameraPos, f32 dt
// 20-23: vec3f volSize, f32 rayDelta
// 24-27: vec3f volSizeNorm, f32 waveOn
// 28-31: vec2f resolution, f32 waveValue, f32 energyFilter
// 32-35: f32 energyMultiplier, f32 waveSourceType, f32 global alpha multiplier, f32 +x projection alpha multiplier
// total 36 * f32 = 144 bytes

const uni = new Uniforms();
uni.addUniform("invMatrix", "mat4x4f");   // inverse proj*view matrix
uni.addUniform("cameraPos", "vec3f");     // camera position in world space
uni.addUniform("dt", "f32");              // simulation time step
uni.addUniform("volSize", "vec3f");       // volume size in voxels
uni.addUniform("rayDtMult", "f32");       // raymarch sampling factor
uni.addUniform("volSizeNorm", "vec3f");   // normalized volume size (volSize / max(volSize))
uni.addUniform("waveOn", "f32");          // whether the wave is on or not
uni.addUniform("resolution", "vec2f");    // canvas resolution: x-width, y-height
uni.addUniform("waveValue", "f32");       // wave value at current time
uni.addUniform("energyFilter", "f32"); // energy filter strength, 0 = off
uni.addUniform("energyMult", "f32");   // energy rendering multiplier
uni.addUniform("waveSourceType", "f32");  // source type: 0=plane, 1=point
uni.addUniform("globalAlpha", "f32");     // global alpha multiplier
uni.addUniform("plusXAlpha", "f32");      // +x face alpha multiplier
uni.addUniform("wavePos", "vec3f");       // Wave beam/point center position
uni.addUniform("waveHalfSize", "vec2f");  // Wave beam size in Y and Z
uni.finalize();

const textures = {
  stateTex0: null,
  stateTex1: null,
  energyTex: null,
  speedTex: null,
};

let time = 0;
let dt = 0.5;
let oldDt;

let dtPerFrame = 1;

let waveOn = true;

let defaultEnergyFilterStrength = 100;
let energyFilterStrength = defaultEnergyFilterStrength;

const waveformOptions = Object.freeze({
  sine: 0,
  square: 1,
  triangle: 2,
  sawtooth: 3
});

const waveSettings = {
  wavelength: 6,
  waveform: waveformOptions.sine,
  amp: 1
}

const sharedSettings = {
  radius: 64,
  refractiveIndex: 1.5,
  thickness: 16,
};

// simulation domain size [x, y, z], ex. [384, 256, 256], [512, 256, 384]
const simulationDomain = [384, 256, 256];//[768, 384, 384];
let newDomainSize = vec3.clone(simulationDomain);
let simVoxelCount = simulationDomain[0] * simulationDomain[1] * simulationDomain[2];

let yMidpt = Math.floor(simulationDomain[1] / 2);
let zMidpt = Math.floor(simulationDomain[2] / 2);

const simulationDomainNorm = simulationDomain.map(v => v / Math.max(...simulationDomain));
let waveSpeedData = new Float32Array(simulationDomain[0] * simulationDomain[1] * simulationDomain[2]).fill(1);

let cleared = false;

const wavePos = [0, yMidpt, zMidpt];
const waveHalfSize = [36, 36];

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
      quadSymmetricFlatPreset(flatPresets[presetType], presetXOffset, barrierThickness, presetSettings[presetType]);
      break;
    case "Lens":
      createLens(lensPresets[gui.io.lensType()], true, presetXOffset, presetSettings.Lens);
      break;
    case "PhasePlate":
      const plateType = gui.io.phasePlateType();
      phasePlate(phasePlatePresets[plateType], presetXOffset, presetSettings[plateType]);
      break;
    case "Prism":
      nGonPrism(presetXOffset, presetSettings.Prism);
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
    { texture: textures.energyTex },
    zeros,
    { offset: 0, bytesPerRow: simulationDomain[0] * 4, rowsPerImage: simulationDomain[1] },
    { width: simulationDomain[0], height: simulationDomain[1], depthOrArrayLayers: simulationDomain[2] },
  );
  uni.values.dt.set([dt]);
  time = 0;
  uni.values.waveValue.set([0]);
}

function hardReset() {
  cancelAnimationFrame(rafId);
  clearInterval(perfIntId);
  if (!vec3.equals(simulationDomain, newDomainSize)) resizeDomain(newDomainSize);
  textures.speedTex.destroy();
  if (cleared) main();
  else main().then(refreshPreset);
  uni.values.dt.set([dt]);
  time = 0;
  uni.values.waveValue.set([0]);
}


const canvas = document.getElementById("canvas");

const gui = new GUI("3D wave sim on WebGPU", canvas);

// Performance section
{
  gui.addGroup("perf", "Performance");
  gui.addStringOutput("res", "Resolution", "", "perf");
  gui.addHalfWidthGroups("perfL", "perfR", "perf");
  gui.addNumericOutput("fps", "FPS", "", 1, "perfL");
  gui.addNumericOutput("frameTime", "Frame", "ms", 2, "perfL");
  gui.addNumericOutput("jsTime", "JS", "ms", 2, "perfL");
  gui.addNumericOutput("computeTime", "Compute", "ms", 2, "perfR");
  gui.addNumericOutput("boundaryTime", "Boundary", "ms", 2, "perfR");
  gui.addNumericOutput("renderTime", "Render", "ms", 2, "perfR");
}

// Camera state section
{
  gui.addGroup("camState", "Camera state");
  gui.addNumericOutput("camFOV", "FOV", "°", 2, "camState");
  gui.addNumericOutput("camDist", "Dst", "", 2, "camState");
  gui.addStringOutput("camTarget", "Tgt", "", "camState");
  gui.addStringOutput("camPos", "Pos", "", "camState");
  gui.addNDimensionalOutput(["camAlt", "camAz"], "Alt/az", "°", ", ", 2, "camState");
}

// Sim controls
{
  gui.addGroup("simCtrl", "Sim controls");
  gui.addNumericInput("dt", true, "dt (reinit)", { min: 0, max: 1, step: 0.01, val: dt, float: 2 }, "simCtrl", (newDt) => {
    if (oldDt) oldDt = newDt;
    else dt = newDt;
  }, "Simulation delta-time; must meet the CFL condition for stability; requires reinitialization to apply");
  gui.addNumericInput("xSize", false, "X size (reinit)", { min: 8, max: 1024, step: 8, val: simulationDomain[0], float: 0 }, "simCtrl", (value) => newDomainSize[0] = value, "Requires reinitialization to apply");
  gui.addNumericInput("ySize", false, "Y size (reinit)", { min: 8, max: 512, step: 8, val: simulationDomain[1], float: 0 }, "simCtrl", (value) => newDomainSize[1] = value, "Requires reinitialization to apply");
  gui.addNumericInput("zSize", false, "Z size (reinit)", { min: 8, max: 512, step: 8, val: simulationDomain[2], float: 0 }, "simCtrl", (value) => newDomainSize[2] = value, "Requires reinitialization to apply");
  gui.addNumericInput("wavelength", true, "Wavelength", { min: 4, max: 100, step: 0.1, val: 6, float: 1 }, "simCtrl", (value) => waveSettings.wavelength = value );
  gui.addNumericInput("amp", true, "Amplitude", { min: 0.1, max: 5, step: 0.1, val: 1, float: 1 }, "simCtrl", (value) => { waveSettings.amp = value; });
  gui.addHalfWidthGroups("waveformOptions", "sourceTypeOptions", "simCtrl");
  gui.addRadioOptions("waveform", ["sine", "square", "triangle", "sawtooth"], "sine", "waveformOptions", {}, (value) => waveSettings.waveform = waveformOptions[value]);
  gui.addNumericInput("wavePX", true, "X pos", { min: 0, max: 1024, step: 16, val: 0, float: 0 }, "simCtrl", (value) => {
    wavePos[0] = value;
    uni.values.wavePos.set(wavePos);
  }, "Wave source X-coordinate");
  gui.addNumericInput("wavePY", true, "Y pos", { min: 0, max: 1, step: 0.01, val: 0.5, float: 2 }, "simCtrl", (value) => {
    wavePos[1] = Math.round(value * simulationDomain[1]);
    uni.values.wavePos.set(wavePos);
  }, "Wave source Y-coordinate normalized to sim domain");
  gui.addNumericInput("wavePZ", true, "Z pos", { min: 0, max: 1, step: 0.01, val: 0.5, float: 2 }, "simCtrl", (value) => {
    wavePos[2] = Math.round(value * simulationDomain[2]);
    uni.values.wavePos.set(wavePos);
  }, "Wave source Z-coordinate normalized to sim domain");
  gui.addNumericInput("waveSY", true, "Y size", { min: 8, max: 256, step: 1, val: 36, float: 0 }, "simCtrl", (value) => {
    waveHalfSize[0] = value / 2;
    uni.values.waveHalfSize.set(waveHalfSize);
  }, "Wave source Y size on the simulation YZ plane");
  gui.addNumericInput("waveSZ", true, "Z size", { min: 8, max: 256, step: 1, val: 36, float: 0 }, "simCtrl", (value) => { waveHalfSize[1] = value / 2; uni.values.waveHalfSize.set(waveHalfSize); }, "Wave source Z size on the simulation YZ plane");
  gui.addRadioOptions("sourceType", ["plane", "beam", "point"], "plane", "sourceTypeOptions", {
    plane: [],
    beam: ["wavePY", "wavePZ", "waveSY", "waveSZ"],
    point: ["wavePX", "wavePY", "wavePZ"],
  }, (value) => uni.values.waveSourceType.set([value === "plane"? 0 : value === "beam" ? 1 : 2]));
  gui.addButton("waveOn", "Toggle wave generator", true, "simCtrl", () => {
    waveOn = !waveOn;
    if (waveOn) {
      uni.values.waveOn.set([1]);
      waveSettings.amp = gui.io.amp.value;
    }
  });
  gui.addButton("toggleSim", "Play / Pause", false, "simCtrl", () => {
    if (oldDt) {
      dt = oldDt;
      oldDt = null;
      uni.values.dt.set([dt]);
    } else {
      oldDt = dt;
      dt = 0;
    }
  });

  gui.addButton("softRestart", "Restart", false, "simCtrl", softReset);
  gui.addButton("hardRestart", "Reinitialize", true, "simCtrl", hardReset);
}

// Preset controls
{
  gui.addGroup("presets", "Presets");

  gui.addRadioOptions("shape", ["circular", "square", "linear"], "circular", "presets", {}, (value) => presetSettings.Aperture.shape = presetSettings.ZonePlate.shape = shapes[value]);

  gui.addNumericInput("f", true, "Focal length", { min: 4, max: 512, step: 1, val: 192, float: 0 }, "presets", (value) => presetSettings.ZonePlate.f = presetSettings.CircularLens.f = value);
  gui.addNumericInput("nCutouts", true, "# Cutouts", { min: 1, max: 20, step: 1, val: 4, float: 0 }, "presets", (value) => presetSettings.ZonePlate.nCutouts = value);

  gui.addNumericInput("slitWidth", true, "Slit width", { min: 3, max: 512, step: 1, val: 8, float: 0 }, "presets", (value) => presetSettings.DoubleSlit.slitWidth = value);
  gui.addNumericInput("slitSpacing", true, "Slit spacing", { min: 0, max: 512, step: 1, val: 32, float: 0 }, "presets", (value) => presetSettings.DoubleSlit.slitSpacing = value);
  gui.addNumericInput("slitHeight", true, "Slit height", { min: 0, max: 512, step: 1, val: 64, float: 0 }, "presets", (value) => presetSettings.DoubleSlit.slitHeight = value);

  gui.addNumericInput("radius", true, "Radius", { min: 0, max: 256, step: 1, val: sharedSettings.radius, float: 0 }, "presets", (value) => sharedSettings.radius = value);

  gui.addCheckbox("invert", "Invert barrier", false, "presets", (checked) => presetSettings.Aperture.invert = checked);

  gui.addRadioOptions("lensType", ["elliptical", "parabolic"], "parabolic", "presets");
  gui.addNumericInput("lensThickness", true, "Thickness", { min: 4, max: 100, step: 1, val: 16, float: 0 }, "presets", (value) =>
    sharedSettings.thickness = value
  );
  gui.addNumericInput("refractiveIndex", false, "Refractive index", { min: 0.5, max: 2, step: 0.01, val: sharedSettings.refractiveIndex, float: 2 }, "presets", (value) => sharedSettings.refractiveIndex = value);
  gui.addNumericInput("halfLens", true, "Half lens", { min: -1, max: 1, step: 1, val: 0, float: 0 }, "presets", (value) => presetSettings.Lens.half = value, "-1: curved toward source, 0: both halves, 1: flat toward source");
  gui.addCheckbox("outerBarrier", "Outer barrier", true, "presets", (checked) => presetSettings.Lens.outerBarrier = checked);


  gui.addNumericInput("barrierThickness", true, "Thickness", { min: 1, max: 16, step: 1, val: 2, float: 0 }, "presets", (value) => barrierThickness = value);
  gui.addNumericInput("xOffset", true, "X Offset", { min: 0, max: 512, step: 1, val: 16, float: 0 }, "presets", (value) => presetXOffset = value);

  gui.addGroup("phasePlateOptions-container", null, null, "presets");
  gui.addNumericInput("nVortices", true, "n vortices", { min: -4, max: 4, step: 1, val: 1, float: 0 }, "phasePlateOptions-container", (value) => presetSettings.Vortex.n = value);
  gui.addNumericInput("exp", true, "exp", { min: 0, max: 5, step: 0.1, val: 2, float: 1 }, "phasePlateOptions-container", (value) => presetSettings.PowerLens.n = value);
  gui.addCheckbox("invertLens", "Invert lens", false, "phasePlateOptions-container", (checked) => presetSettings.PowerLens.invert = presetSettings.CircularLens.invert = checked);

  gui.addRadioOptions("phasePlateType", ["Vortex", "PowerLens", "CircularLens"], "Vortex", "phasePlateOptions-container", {
    "Vortex": ["nVortices"],
    "PowerLens": ["exp", "invertLens"],
    "CircularLens": ["f", "invertLens"],
  });

  gui.addNumericInput("nSides", true, "n sides", { min: 3, max: 24, step: 1, val: 3, float: 0 }, "presets", (value) => presetSettings.Prism.n = value);
  gui.addNumericInput("rotation", true, "Rotation", { min: 0, max: 360, step: 1, val: 0, float: 0 }, "presets", (value) => presetSettings.Prism.rot = Math.PI / 180 * value);

  gui.addDropdown("presetSelect", "Select preset", ["ZonePlate", "DoubleSlit", "Aperture", "Lens", "PhasePlate", "Prism"], "presets", {
    "ZonePlate": ["shape", "f", "nCutouts", "barrierThickness"],
    "DoubleSlit": ["slitWidth", "slitSpacing", "slitHeight", "barrierThickness"],
    "Aperture": ["shape", "radius", "invert", "barrierThickness"],
    "Lens": ["radius", "lensType", "lensThickness", "refractiveIndex", "halfLens", "outerBarrier"],
    "PhasePlate": ["radius", "refractiveIndex", "lensThickness", "phasePlateOptions"],
    "Prism": ["radius", "refractiveIndex", "rotation", "nSides"]
  });

  gui.addButton("updatePreset", "Load preset", false, "presets", () => refreshPreset(false));
  gui.addButton("clearUpdatePreset", "Clear & load", false, "presets", () => refreshPreset(true));
  gui.addButton("clearPreset", "Clear", true, "presets", () => updateSpeedTexture(true));
}

// Visualization controls
{
  gui.addGroup("visCtrl", "Visualization controls");
  gui.addNumericInput("globalAlpha", true, "Global alpha", { min: 0.1, max: 5, step: 0.1, val: 2, float: 1 }, "visCtrl", (value) => uni.values.globalAlpha.set([value]), "Global alpha multiplier");
  gui.addNumericInput("rayDtMult", true, "Ray dt mult", { min: 0.1, max: 5, step: 0.1, val: 2, float: 1 }, "visCtrl", (value) => uni.values.rayDtMult.set([value]), "Raymarching step multipler; higher has better visual quality, lower has better performance");
  gui.addCheckbox("energy", "Visualize energy", true, "visCtrl", (checked) => {
    energyFilterStrength = checked ? defaultEnergyFilterStrength : 0;
    uni.values.energyFilter.set([energyFilterStrength]);
  });
  gui.addNumericInput("plusXAlpha", true, "+X energy a", { min: 1, max: 5, step: 0.1, val: 2, float: 1 }, "visCtrl", (value) => uni.values.plusXAlpha.set([value]), "+X energy projection alpha multiplier");
  gui.addNumericInput("energyMult", true, "Energy mult", { min: 0.01, max: 5, step: 0.01, val: 1, float: 2 }, "visCtrl", (value) => uni.values.energyMult.set([value]), "Raw energy value multiplier before transfer function");
  gui.addNumericInput("energyFilter", true, "Energy filter", { min: 0, max: 3, step: 0.1, val: 2, float: 1 }, "visCtrl", (value) => {
    value = Math.pow(10, value);
    defaultEnergyFilterStrength = value;
    energyFilterStrength = gui.io.energy.checked ? defaultEnergyFilterStrength : 0;
    uni.values.energyFilter.set([value]);
  }, "Energy low pass filter strength");
}

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
gui.addGroup("guiControls", "GUI controls", `
  <div>
    Click on section titles to expand/collapse
    <br>
    Hover on input labels for more info if applicable
    <br>
    Click to toggle between raw number and slider type input
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
  uni.values.resolution.set([canvas.width, canvas.height]);
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