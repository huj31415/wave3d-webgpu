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
uni.ampValue = uniformValues.subarray(kWaveSettingsOffset, kWaveSettingsOffset + 2);
uni.intensityFilterValue = uniformValues.subarray(kIntensityFilterOffset, kIntensityFilterOffset + 1);
uni.intensityMultValue = uniformValues.subarray(kIntensityMultOffset, kIntensityMultOffset + 1);


let dt = 0.5;
let oldDt;

let defaultIntensityFilterStrength = 50;
let intensityFilterStrength = defaultIntensityFilterStrength;

let amp = 1, wavelength = 6;

// simulation domain size [x, y, z], ex. [384, 256, 256], [512, 256, 384]
const simulationDomain = [384, 256, 256];

let yMidpt = Math.floor(simulationDomain[1] / 2);
let zMidpt = Math.floor(simulationDomain[2] / 2);

const simulationDomainNorm = simulationDomain.map(v => v / Math.max(...simulationDomain));
const waveSpeedData = new Float32Array(simulationDomain[0] * simulationDomain[1] * simulationDomain[2]).fill(1);

let timeBuffer;

const canvas = document.getElementById("canvas");


const gui = new GUI("3D wave sim on WebGPU", canvas);

gui.addGroup("perf", "Performance");
gui.addStringOutput("res", "Resolution", "", "perf");
gui.addHalfWidthGroups("perfL", "perfR", "perf");
gui.addNumericOutput("fps", "FPS", "", 1, "perfL");
gui.addNumericOutput("frameTime", "Frame", "", 2, "perfL");
gui.addNumericOutput("jsTime", "JS", "", 2, "perfL");
gui.addNumericOutput("computeTime", "Compute", "", 2, "perfR");
gui.addNumericOutput("renderTime", "Render", "", 2, "perfR");

gui.addGroup("camState", "Camera state");
gui.addNumericOutput("camFOV", "FOV", "°", 2, "camState");
gui.addNumericOutput("camDist", "Dst", "", 2, "camState");
gui.addStringOutput("camTarget", "Tgt", "", "camState");
gui.addStringOutput("camPos", "Pos", "", "camState");
gui.addNDimensionalOutput(["camAlt", "camAz"], "Alt/az", "°", ", ", 2, "camState");

gui.addGroup("simCtrl", "Sim controls");
gui.addNumericInput("dt", true, "dt", 0, 1, 0.01, 0.5, 2, "simCtrl", (value) => {
  const newDt = value;
  if (oldDt) oldDt = newDt;
  else dt = newDt;
});
gui.addButton("toggleSim", "Play / Pause", false, "simCtrl", () => {
  if (oldDt) {
    dt = oldDt;
    oldDt = null;
  } else {
    oldDt = dt;
    dt = 0;
  }
});
gui.addButton("restartSim", "Restart", false, "simCtrl", () => {
  cancelAnimationFrame(rafId);
  clearInterval(intId);
  device.queue.writeBuffer(timeBuffer, 0, new Float32Array([0]));
  // device.destroy();
  // device = null;
  main();
});

gui.addGroup("visCtrl", "Visualization controls");
gui.addCheckbox("intensity", "Visualize intensity", true, "visCtrl", (checked) => {
  intensityFilterStrength = checked ? defaultIntensityFilterStrength : 0;
  uni.intensityFilterValue.set([intensityFilterStrength]);
});

gui.addGroup("camKeybinds", "Camera controls",
  `<div>
    Orbit: leftclick / arrows
    <br>
    Pan: rightclick / wasd
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
let rafId, intId;


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
const randMax = (max) => Math.random() * max;

const index3d = (x, y, z) => {
  return x + simulationDomain[0] * (y + z * simulationDomain[1]);
}

function assert(cond, msg = '') {
  if (!cond) {
    throw new Error(msg);
  }
}

// We track command buffers so we can generate an error if
// we try to read the result before the command buffer has been executed.
const s_unsubmittedCommandBuffer = new Set();

/* global GPUQueue */
GPUQueue.prototype.submit = (function (origFn) {
  return function (commandBuffers) {
    origFn.call(this, commandBuffers);
    commandBuffers.forEach(cb => s_unsubmittedCommandBuffer.delete(cb));
  };
})(GPUQueue.prototype.submit);

// See https://webgpufundamentals.org/webgpu/lessons/webgpu-timing.html
class TimingHelper {
  #canTimestamp;
  #device;
  #querySet;
  #resolveBuffer;
  #resultBuffer;
  #commandBuffer;
  #resultBuffers = [];
  // state can be 'free', 'need resolve', 'wait for result'
  #state = 'free';

  constructor(device) {
    this.#device = device;
    this.#canTimestamp = device.features.has('timestamp-query');
    if (this.#canTimestamp) {
      this.#querySet = device.createQuerySet({
        type: 'timestamp',
        count: 2,
      });
      this.#resolveBuffer = device.createBuffer({
        size: this.#querySet.count * 8,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
    }
  }

  #beginTimestampPass(encoder, fnName, descriptor) {
    if (this.#canTimestamp) {
      assert(this.#state === 'free', 'state not free');
      this.#state = 'need resolve';

      const pass = encoder[fnName]({
        ...descriptor,
        ...{
          timestampWrites: {
            querySet: this.#querySet,
            beginningOfPassWriteIndex: 0,
            endOfPassWriteIndex: 1,
          },
        },
      });

      const resolve = () => this.#resolveTiming(encoder);
      const trackCommandBuffer = (cb) => this.#trackCommandBuffer(cb);
      pass.end = (function (origFn) {
        return function () {
          origFn.call(this);
          resolve();
        };
      })(pass.end);

      encoder.finish = (function (origFn) {
        return function () {
          const cb = origFn.call(this);
          trackCommandBuffer(cb);
          return cb;
        };
      })(encoder.finish);

      return pass;
    } else {
      return encoder[fnName](descriptor);
    }
  }

  beginRenderPass(encoder, descriptor = {}) {
    return this.#beginTimestampPass(encoder, 'beginRenderPass', descriptor);
  }

  beginComputePass(encoder, descriptor = {}) {
    return this.#beginTimestampPass(encoder, 'beginComputePass', descriptor);
  }

  #trackCommandBuffer(cb) {
    if (!this.#canTimestamp) {
      return;
    }
    assert(this.#state === 'need finish', 'you must call encoder.finish');
    this.#commandBuffer = cb;
    s_unsubmittedCommandBuffer.add(cb);
    this.#state = 'wait for result';
  }

  #resolveTiming(encoder) {
    if (!this.#canTimestamp) {
      return;
    }
    assert(
      this.#state === 'need resolve',
      'you must use timerHelper.beginComputePass or timerHelper.beginRenderPass',
    );
    this.#state = 'need finish';

    this.#resultBuffer = this.#resultBuffers.pop() || this.#device.createBuffer({
      size: this.#resolveBuffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    encoder.resolveQuerySet(this.#querySet, 0, this.#querySet.count, this.#resolveBuffer, 0);
    encoder.copyBufferToBuffer(this.#resolveBuffer, 0, this.#resultBuffer, 0, this.#resultBuffer.size);
  }

  async getResult() {
    if (!this.#canTimestamp) {
      return 0;
    }
    assert(
      this.#state === 'wait for result',
      'you must call encoder.finish and submit the command buffer before you can read the result',
    );
    assert(!!this.#commandBuffer); // internal check
    assert(
      !s_unsubmittedCommandBuffer.has(this.#commandBuffer),
      'you must submit the command buffer before you can read the result',
    );
    this.#commandBuffer = undefined;
    this.#state = 'free';

    const resultBuffer = this.#resultBuffer;
    await resultBuffer.mapAsync(GPUMapMode.READ);
    const times = new BigInt64Array(resultBuffer.getMappedRange());
    const duration = Number(times[1] - times[0]);
    resultBuffer.unmap();
    this.#resultBuffers.push(resultBuffer);
    return duration;
  }
}