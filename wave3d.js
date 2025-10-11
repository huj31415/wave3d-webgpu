let adapter, device;
let gpuInfo = false;

async function main() {

  if (device) device.destroy();

  // let maxComputeInvocationsPerWorkgroup, maxBufferSize, f32filterable;

  // WebGPU Setup
  // if (!device) {
  adapter = await navigator.gpu?.requestAdapter();

  const maxComputeInvocationsPerWorkgroup = adapter.limits.maxComputeInvocationsPerWorkgroup;
  const maxBufferSize = adapter.limits.maxBufferSize;
  const f32filterable = adapter.features.has("float32-filterable");

  // compute workgroup size 16*8*8 | 32*8*4 | 64*4*4 = 1024 threads if maxComputeInvocationsPerWorkgroup >= 1024, otherwise 16*4*4 = 256 threads
  const largeWg = maxComputeInvocationsPerWorkgroup >= 1024;
  const [wg_x, wg_y, wg_z] = largeWg ? [16, 8, 8] : [16, 4, 4];

  if (!gpuInfo) {
    gui.addGroup("deviceInfo", "Device info", `
<pre><span ${!largeWg ? "class='warn'" : ""}>maxComputeInvocationsPerWorkgroup: ${maxComputeInvocationsPerWorkgroup}
workgroup: [${wg_x}, ${wg_y}, ${wg_z}]</span>
maxBufferSize: ${maxBufferSize}
f32filterable: ${f32filterable}
</pre>
    `);
    gpuInfo = true;
  }

  device = await adapter?.requestDevice({
    requiredFeatures: [
      (adapter.features.has("timestamp-query") ? "timestamp-query" : ""),
      (f32filterable ? "float32-filterable" : ""),
    ],
    requiredLimits: {
      maxComputeInvocationsPerWorkgroup: maxComputeInvocationsPerWorkgroup,
      maxBufferSize: maxBufferSize,
    }
  });
  device.addEventListener('uncapturederror', event => {
    if (event.error.message.includes("max buffer size limit"))
      alert(`Max buffer size exceeded. Your device supports max size ${maxBufferSize}, specified size ${simVoxelCount() * 4}`);
    // else alert(msg);
  });

  // restart if device crashes
  device.lost.then((info) => {
    if (info.reason != "destroyed") {
      hardReset();
      console.warn("WebGPU device lost, reinitializing.");
    }
  });

  // }
  if (!device) {
    alert("Browser does not support WebGPU");
    document.body.textContent = "WebGPU is not supported in this browser.";
    return;
  }
  const context = canvas.getContext("webgpu");
  const swapChainFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device: device,
    format: swapChainFormat,
  });

  // texture layout: r32float
  // 1: texture_storage_3d r32float<rw> past/future
  // 2: texture_storage_3d r32float<r> present
  // 3: texture_storage_3d r32float<r> speed
  // render 1 (new future), then switch 1 and 2 so that old present becomes past and old future becomes present

  const newTexture = (name) => device.createTexture({
    size: simulationDomain,
    dimension: "3d",
    format: "r32float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
    label: `${name} texture`
  });

  textures.stateTex0 = newTexture("state0");
  textures.stateTex1 = newTexture("state1");
  textures.energyTex = newTexture("energy");
  textures.speedTex = newTexture("wavespeed");
  updateSpeedTexture();

  const uniformBuffer = uni.createBuffer(device);

  const newComputePipeline = (shaderCode, name) =>
    device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({
          code: shaderCode,
          label: `${name} compute module`
        }),
        entryPoint: 'main'
      },
      label: `${name} compute pipeline`
    });

  const waveComputePipeline = newComputePipeline(waveShaderCode(wg_x, wg_y, wg_z), "wave");

  const waveComputeBindGroup = (tex0, tex1) => device.createBindGroup({
    layout: waveComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: tex0.createView() },
      { binding: 2, resource: tex1.createView() },
      { binding: 3, resource: textures.speedTex.createView() },
      { binding: 4, resource: textures.energyTex.createView() },
    ],
    label: "wave compute bind group"
  });

  const waveComputeBindGroups = [
    waveComputeBindGroup(textures.stateTex0, textures.stateTex1),
    waveComputeBindGroup(textures.stateTex1, textures.stateTex0)
  ];

  const boundaryComputePipeline = newComputePipeline(boundaryShaderCode(wg_x, wg_y, wg_z), "boundary");

  const boundaryComputeBindGroup = (tex0, tex1) => device.createBindGroup({
    layout: boundaryComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: tex0.createView() },
      { binding: 2, resource: tex1.createView() },
      { binding: 3, resource: textures.speedTex.createView() }
    ],
    label: "boundary compute bind group"
  });

  const boundaryComputeBindGroups = [
    boundaryComputeBindGroup(textures.stateTex0, textures.stateTex1),
    boundaryComputeBindGroup(textures.stateTex1, textures.stateTex0)
  ];

  const renderModule = device.createShaderModule({
    code: renderShaderCode,
    label: "render module"
  });

  const filter = f32filterable ? "linear" : "nearest";
  const sampler = device.createSampler({
    magFilter: filter,
    minFilter: filter,
  });

  const renderPipeline = device.createRenderPipeline({
    label: '3d volume rendering pipeline',
    layout: 'auto',
    vertex: { module: renderModule },
    fragment: {
      module: renderModule,
      targets: [{ format: swapChainFormat }],
    }
  });

  const renderBindGroup = (tex) => device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: tex.createView() },
      { binding: 2, resource: textures.speedTex.createView() },
      { binding: 3, resource: sampler },
    ],
  });

  const renderBindGroups = [
    renderBindGroup(textures.stateTex1),
    renderBindGroup(textures.stateTex0),
    renderBindGroup(textures.energyTex)
  ];

  const renderPassDescriptor = {
    label: 'render pass',
    colorAttachments: [
      {
        clearValue: [0, 0, 0, 1],
        loadOp: 'clear',
        storeOp: 'store',
      },
    ]
  };
  const filterStrength = 50;

  const waveComputeTimingHelper = new TimingHelper(device);
  const boundaryComputeTimingHelper = new TimingHelper(device);
  const renderTimingHelper = new TimingHelper(device);

  const wgDispatchSize = [
    Math.ceil(simulationDomain[0] / wg_x),
    Math.ceil(simulationDomain[1] / wg_y),
    Math.ceil(simulationDomain[2] / wg_z)
  ]

  // let [tex0, tex1] = [textures.stateTex0, textures.stateTex1];
  let pingPongIndex = 0;

  function render() {
    const startTime = performance.now();
    deltaTime += Math.min(startTime - lastFrameTime - deltaTime, 1e4) / filterStrength;
    const speedMultiplier = Math.min(deltaTime, 50);
    fps += (1e3 / deltaTime - fps) / filterStrength;
    lastFrameTime = startTime;

    if (keyOrbit) {
      const speed = KEY_ROT_SPEED * speedMultiplier;
      camera.orbit(
        (keyState.orbit.left - keyState.orbit.right) * speed,
        (keyState.orbit.up - keyState.orbit.down) * speed
      );
    }
    if (keyPan) {
      const speed = KEY_PAN_SPEED * speedMultiplier;
      camera.pan(
        (keyState.pan.left - keyState.pan.right) * speed,
        (keyState.pan.up - keyState.pan.down) * speed,
        (keyState.pan.forward - keyState.pan.backward) * speed
      );
    }
    if (keyZoom) {
      camera.zoom((keyState.zoom.out - keyState.zoom.in) * KEY_ZOOM_SPEED * speedMultiplier);
    }
    if (keyFOV) {
      camera.adjFOV((keyState.zoom.out - keyState.zoom.in) * KEY_FOV_SPEED * speedMultiplier);
    }
    if (keyFOVWithoutZoom) {
      camera.adjFOVWithoutZoom((keyState.zoom.out - keyState.zoom.in) * KEY_FOV_SPEED * speedMultiplier);
    }

    const canvasTexture = context.getCurrentTexture();
    renderPassDescriptor.colorAttachments[0].view = canvasTexture.createView();

    // device.queue.writeBuffer(uniformBuffer, 0, uni.uniformData);
    uni.update(device.queue);

    const encoder = device.createCommandEncoder();

    const run = dt > 0;

    if (run) {
      if (!waveOn && waveSettings.amp > 0) {
        // ease wave off
        waveSettings.amp -= 0.2 * waveSettings.amp;
        if (waveSettings.amp <= 5e-2) {
          waveSettings.amp = 0;
          uni.values.waveOn.set([0]);
          time = 0;
        }
      }
      if (waveOn || waveSettings.amp > 0) {
        let wave = waveSettings.amp;
        let wavelengthTime = (time += dt) / waveSettings.wavelength;
        switch (waveSettings.waveform) {
          case waveformOptions.sine:
            wave *= Math.sin(Math.PI * 2 * wavelengthTime);
            break;
          case waveformOptions.square:
            wave *= 4 * Math.floor(wavelengthTime) - 2 * Math.floor(2 * wavelengthTime) + 1;
            break;
          case waveformOptions.triangle:
            wave *= 4 * Math.abs(wavelengthTime - Math.floor(wavelengthTime + 0.75) + 0.25) - 1;
            break;
          case waveformOptions.sawtooth:
            wave *= 2 * (wavelengthTime - Math.floor(wavelengthTime + 0.5));
            break;
          default:
            wave *= 0;
        }
        uni.values.waveValue.set([wave]);
      }

      const waveComputePass = waveComputeTimingHelper.beginComputePass(encoder);
      waveComputePass.setPipeline(waveComputePipeline);
      waveComputePass.setBindGroup(0, waveComputeBindGroups[pingPongIndex]);
      waveComputePass.dispatchWorkgroups(...wgDispatchSize);
      waveComputePass.end();

      const boundaryComputePass = boundaryComputeTimingHelper.beginComputePass(encoder);
      boundaryComputePass.setPipeline(boundaryComputePipeline);
      boundaryComputePass.setBindGroup(0, boundaryComputeBindGroups[pingPongIndex]);
      boundaryComputePass.dispatchWorkgroups(...wgDispatchSize);
      boundaryComputePass.end();

      pingPongIndex = 1 - pingPongIndex;
      // [tex0, tex1] = [tex1, tex0]; // swap ping pong textures
    }

    const renderPass = renderTimingHelper.beginRenderPass(encoder, renderPassDescriptor);
    renderPass.setPipeline(renderPipeline);
    renderPass.setBindGroup(0, renderBindGroups[energyFilterStrength > 0 ? 2 : pingPongIndex]);
    renderPass.draw(3, 1, 0, 0);
    renderPass.end();

    const commandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);

    if (run) {
      waveComputeTimingHelper.getResult().then(gpuTime => waveComputeTime += (gpuTime / 1e6 - waveComputeTime) / filterStrength);
      boundaryComputeTimingHelper.getResult().then(gpuTime2 => boundaryComputeTime += (gpuTime2 / 1e6 - boundaryComputeTime) / filterStrength);
    } else {
      waveComputeTime = boundaryComputeTime = 0;
    }
    renderTimingHelper.getResult().then(gpuTime => renderTime += (gpuTime / 1e6 - renderTime) / filterStrength);

    jsTime += (performance.now() - startTime - jsTime) / filterStrength;

    rafId = requestAnimationFrame(render);
  }

  perfIntId = setInterval(() => {
    gui.io.fps(fps);
    gui.io.jsTime(jsTime);
    gui.io.frameTime(deltaTime);
    gui.io.computeTime(waveComputeTime);
    gui.io.boundaryTime(boundaryComputeTime);
    gui.io.renderTime(renderTime);
  }, 100);

  camera.updatePosition();

  uni.values.dt.set([dt]);
  uni.values.volSize.set(simulationDomain);
  uni.values.volSizeNorm.set(simulationDomainNorm);
  uni.values.waveOn.set([1]);
  uni.values.rayDtMult.set([2]);
  uni.values.resolution.set([canvas.width, canvas.height]);
  uni.values.energyFilter.set([energyFilterStrength]);
  uni.values.energyMult.set([1]);
  uni.values.waveSourceType.set([0]);
  uni.values.globalAlpha.set([2]);
  uni.values.plusXAlpha.set([2]);
  uni.values.wavePos.set(wavePos);
  uni.values.waveHalfSize.set(waveHalfSize);
  time = 0;

  rafId = requestAnimationFrame(render);
}

const camera = new Camera(defaults);

main().then(() => quadSymmetricFlatPreset(flatPresets.ZonePlate, presetXOffset, barrierThickness, presetSettings.ZonePlate));