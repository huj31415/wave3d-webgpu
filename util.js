let dt = 0.5;
let oldDt;

// simulation domain size [x, y, z]
const simulationDomain = [512, 256, 384];
const simulationDomainNorm = simulationDomain.map(v => v / Math.max(...simulationDomain));
const waveSpeedData = new Float32Array(simulationDomain[0] * simulationDomain[1] * simulationDomain[2]).fill(1);

let timeBuffer;

const canvas = document.getElementById("canvas");

const uiIDs = [
  "controls",
  "toggleSettings",
  "res",
  "dt",
  "dtValue",
  "G",
  "GValue",
  "jsTime",
  "frameTime",
  "fps",
  "computeTime",
  "renderTime",
  "nBodies",
  "camFOV",
  "camDist",
  "camTarget",
  "camPos",
  "camAlt",
  "camAz",
  "toggleSim",
  "restartSim",
  "export",
  "import",
  "bufferInput"
];

const ui = {};

uiIDs.forEach((id) => ui[id] = document.getElementById(id));
Object.freeze(ui);

ui.dt.addEventListener("input", (event) => {
  const val = parseFloat(event.target.value);
  ui.dtValue.textContent = val.toFixed(2);
  const newDt = event.target.value;
  if (oldDt) oldDt = newDt;
  else dt = newDt;

  uni.dtValue.set([dt]);
});

ui.toggleSim.addEventListener("click", () => {
  if (oldDt) {
    dt = oldDt;
    oldDt = null;
  } else {
    oldDt = dt;
    dt = 0;
  }
});

// requestAnimationFrame id, fps update id
let rafId, intId;

ui.restartSim.addEventListener("click", () => {
  cancelAnimationFrame(rafId);
  clearInterval(intId);
  device.queue.writeBuffer(timeBuffer, 0, new Float32Array([0]));
  // device.destroy();
  // device = null;
  main();
});

ui.toggleSettings.addEventListener("click", () => {
  ui.toggleSettings.innerText = ui.toggleSettings.innerText === ">" ? "<" : ">";
  if (ui.controls.classList.contains("hidden")) {
    ui.controls.classList.remove("hidden");
    ui.toggleSettings.classList.remove("inactive");
  } else {
    ui.controls.classList.add("hidden");
    ui.toggleSettings.classList.add("inactive");
  }
});

// timing
let jsTime = 0, lastFrameTime = performance.now(), deltaTime = 10, fps = 0,
  waveComputeTime = 0, boundaryComputeTime = 0, renderTime = 0;

window.onresize = window.onload = () => {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  camera.updateMatrix();
  uni.resValue.set([canvas.width, canvas.height]);
  ui.res.textContent = [window.innerWidth, window.innerHeight];
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