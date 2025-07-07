let adapter, device;

// uniform layout
// 00-15: mat4x4f inv proj*view
// 16-19: vec3f cameraPos, f32 dt
// 20-23: vec3f volSize, f32 rayDelta
// 24-27: vec3f volSizeNorm, f32 padding
// 28-31: vec2f resolution, vec2f padding
// total 32 * f32 = 128 bytes

// 2*f32 amplitude and wavelength

const uniformStruct = `
    struct Uniforms {
      invMatrix: mat4x4f, // inverse proj*view matrix
      cameraPos: vec3f,   // camera position in world space
      dt: f32,            // simulation time step
      volSize: vec3f,     // volume size in voxels
      rayDtMult: f32,     // raymarch sampling factor
      volSizeNorm: vec3f, // normalized volume size (volSize / max(volSize))
      resolution: vec2f,  // canvas resolution: x-width, y-height
    };
  `;

const uniformValues = new Float32Array(32);

const kMatrixOffset = 0;
const kCamPosOffset = 16;
const kDtOffset = 19;
const kVolSizeOffset = 20;
const kRayDtMultOffset = 23;
const kVolSizeNormOffset = 24;
const kResOffset = 28;

const uni = {};

uni.matrixValue = uniformValues.subarray(kMatrixOffset, kMatrixOffset + 16);
uni.cameraPosValue = uniformValues.subarray(kCamPosOffset, kCamPosOffset + 3);
uni.dtValue = uniformValues.subarray(kDtOffset, kDtOffset + 1);
uni.volSizeValue = uniformValues.subarray(kVolSizeOffset, kVolSizeOffset + 3);
uni.rayDtMultValue = uniformValues.subarray(kRayDtMultOffset, kRayDtMultOffset + 1);
uni.volSizeNormValue = uniformValues.subarray(kVolSizeNormOffset, kVolSizeNormOffset + 3);
uni.resValue = uniformValues.subarray(kResOffset, kResOffset + 2);

async function main() {

  let maxComputeInvocationsPerWorkgroup, maxBufferSize, f32filterable;

  // WebGPU Setup
  if (!device) {
    adapter = await navigator.gpu?.requestAdapter();

    maxComputeInvocationsPerWorkgroup = adapter.limits.maxComputeInvocationsPerWorkgroup;
    maxBufferSize = adapter.limits.maxBufferSize;
    f32filterable = adapter.features.has("float32-filterable");

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
  }
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

  // texture layout: r32float if float32-filterable available, else r16float
  // 1: texture_storage_3d r32float<rw> past/future
  // 2: texture_storage_3d r32float<r> present
  // 3: texture_storage_3d r32float<r> speed
  // render 1 (new future), then switch 1 and 2 so that old present becomes past and old future becomes present

  const stateTex0 = device.createTexture({
    size: simulationDomain,
    dimension: "3d",
    format: "r32float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
    label: "state texture 0",
  });
  const stateTex1 = device.createTexture({
    size: simulationDomain,
    dimension: "3d",
    format: "r32float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
    label: "state texture 1",
  });
  const projectionTex = device.createTexture({
    size: simulationDomain,
    dimension: "3d",
    format: "r32float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
    label: "projection texture",
  });
  const speedTex = device.createTexture({
    size: simulationDomain,
    dimension: "3d",
    format: "r32float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
    label: "wavespeed texture",
  });

  
  quadSymmetricFlatBarrier(symmetricPresets.circleAperture, 64, 2, [32]);

  device.queue.writeTexture(
    { texture: speedTex },
    waveSpeedData,
    { offset: 0, bytesPerRow: simulationDomain[0] * 4, rowsPerImage: simulationDomain[1] },
    { width: simulationDomain[0], height: simulationDomain[1], depthOrArrayLayers: simulationDomain[2] },
  );

  const uniformBuffer = device.createBuffer({
    size: uniformValues.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });


  // time for wave generator
  timeBuffer = device.createBuffer({
    size: Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: "timeBuffer"
  });

  // compute workgroup size 16*8*8 | 32*8*4 | 64*4*4 = 1024 threads if maxComputeInvocationsPerWorkgroup >= 1024, otherwise 16*4*4 = 256 threads
  const [wg_x, wg_y, wg_z] = maxComputeInvocationsPerWorkgroup >= 1024 ? [16, 8, 8] : [16, 4, 4];

  const waveComputeModule = device.createShaderModule({
    code: `
      ${uniformStruct}

      @group(0) @binding(0) var<uniform> uni: Uniforms;
      @group(0) @binding(1) var past_future:  texture_storage_3d<r32float, read_write>;
      @group(0) @binding(2) var present:      texture_storage_3d<r32float, read>;
      @group(0) @binding(3) var waveSpeed:    texture_storage_3d<r32float, read>;
      @group(0) @binding(4) var<storage, read_write> time: f32;

      const WG_X: u32 = ${wg_x};
      const WG_Y: u32 = ${wg_y};
      const WG_Z: u32 = ${wg_z};

      const directions: array<vec3i, 6> = array<vec3i, 6>(
        vec3i(-1, 0, 0), // left
        vec3i( 1, 0, 0), // right
        vec3i(0, -1, 0), // down
        vec3i(0,  1, 0), // up
        vec3i(0, 0, -1), // back
        vec3i(0, 0,  1)  // front
      );

      // var<workgroup> presentTile: array<f32, (WG_X + 2) * (WG_Y + 2) * (WG_Z + 2)>;

      // fn localPresentIndex(idx: vec3u) -> u32 {
      //   return idx.x + (WG_X + 2u) * (idx.y + (WG_Y + 2u) * idx.z);
      // }

      // 3d wave compute shader
      @compute @workgroup_size(WG_X, WG_Y, WG_Z)
      fn main(
        @builtin(global_invocation_id) gid: vec3u
      ) {
        if (all(gid == vec3u(0))) { time += uni.dt; }

        let volSize = vec3u(uni.volSize);
        let gid_i = vec3i(gid);

        // check if the index is within bounds
        if (any(gid >= volSize)) { return; }

        // read the states
        let pastValue = textureLoad(past_future, gid).r;
        let presentValue = textureLoad(present, gid).r;

        let adjIndex = array<vec3i, 6>(
          gid_i + directions[0], // left
          gid_i + directions[1], // right
          gid_i + directions[2], // down
          gid_i + directions[3], // up
          gid_i + directions[4], // back
          gid_i + directions[5]  // front
        );

        // 7 point stencil for laplacian
        let adjArrayPresent = array<f32, 6>(
          textureLoad(present, adjIndex[0]).r, // left
          textureLoad(present, adjIndex[1]).r, // right
          textureLoad(present, adjIndex[2]).r, // down
          textureLoad(present, adjIndex[3]).r, // up
          textureLoad(present, adjIndex[4]).r, // back
          textureLoad(present, adjIndex[5]).r  // front
        );

        var laplacian = -6.0 * presentValue;

        for (var i = 0; i < 6; i++) {
          laplacian += adjArrayPresent[i];
        }

        // read the wave speed
        let cdt = textureLoad(waveSpeed, gid).r * uni.dt;
        if (cdt <= 0) {
          textureStore(past_future, gid, vec4f(0));
          return;
        }

        // compute the new value based on the wave equation
        var newValue = 2 * presentValue - pastValue + cdt * cdt * laplacian;

        // Wave generators
        let waveGen = 1 * sin(time * 6.28f / 6);
        // plane wave
        if (gid.x == 2) {
          newValue = waveGen;
        }
        // point source
        // if (all(gid == vec3u(18, volSize.y / 2, volSize.z / 2))) {
        //   newValue = 100 * waveGen;
        // }
        
        // if (uniforms.waveOn == 1) {
        //   let waveGen = uniforms.amp * sin(time * 6.28f / uniforms.wavelength);
        //   if (uniforms.waveType == 0 && x == 1) {
        //     // plane wave
        //     stateNext[index] = waveGen;
        //   } else if (uniforms.waveType == 1 && x == 50 && y == height / 2) {
        //     // point source
        //     stateNext[index] = 10 * waveGen;
        //   }
        // }
        
        // write to the past/future texture
        textureStore(past_future, gid, vec4f(newValue, 0.0, 0.0, 0.0));
      }
    `,
    label: "wave compute module"
  });

  const waveComputePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: waveComputeModule, entryPoint: 'main' },
    label: "wave compute pipeline"
  });

  const waveComputeBindGroup = (tex0, tex1) => device.createBindGroup({
    layout: waveComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: tex0.createView() },
      { binding: 2, resource: tex1.createView() },
      { binding: 3, resource: speedTex.createView() },
      { binding: 4, resource: { buffer: timeBuffer } }
    ],
    label: "wave compute bind group"
  });

  const boundaryComputeModule = device.createShaderModule({
    code: `
      ${uniformStruct}

      @group(0) @binding(0) var<uniform> uni: Uniforms;
      @group(0) @binding(1) var future:       texture_storage_3d<r32float, read_write>;
      @group(0) @binding(2) var present:      texture_storage_3d<r32float, read>;
      @group(0) @binding(3) var waveSpeed:    texture_storage_3d<r32float, read>;

      const WG_X: u32 = ${wg_x};
      const WG_Y: u32 = ${wg_y};
      const WG_Z: u32 = ${wg_z};

      const directions: array<vec3i, 6> = array<vec3i, 6>(
        vec3i(-1, 0, 0), // left
        vec3i( 1, 0, 0), // right
        vec3i(0, -1, 0), // down
        vec3i(0,  1, 0), // up
        vec3i(0, 0, -1), // back
        vec3i(0, 0,  1)  // front
      );

      fn mur1stOrder(gid_i: vec3i, index: u32, frac: f32) -> f32 {
        let idx = gid_i + directions[index];
        return textureLoad(present, idx).r + frac * (textureLoad(future, idx).r - textureLoad(future, gid_i).r);
      }

      // 3d wave compute shader
      @compute @workgroup_size(WG_X, WG_Y, WG_Z)
      fn main(
        @builtin(global_invocation_id) gid: vec3u
      ) {

        let volSize = vec3u(uni.volSize);
        let gid_i = vec3i(gid);

        // check if the index is within bounds
        if (any(gid >= volSize)) { return; }

        let adjIndex = array<vec3i, 6>(
          gid_i + directions[0], // left
          gid_i + directions[1], // right
          gid_i + directions[2], // down
          gid_i + directions[3], // up
          gid_i + directions[4], // back
          gid_i + directions[5]  // front
        );

        let adjSpeeds = array<f32, 6>(
          textureLoad(waveSpeed, adjIndex[0]).r, // left
          textureLoad(waveSpeed, adjIndex[1]).r, // right
          textureLoad(waveSpeed, adjIndex[2]).r, // down
          textureLoad(waveSpeed, adjIndex[3]).r, // up
          textureLoad(waveSpeed, adjIndex[4]).r, // back
          textureLoad(waveSpeed, adjIndex[5]).r  // front
        );

        // read the wave speed
        let cdt = textureLoad(waveSpeed, gid).r * uni.dt;
        if (cdt <= 0) { return; }
        let frac = (cdt - 1) / (cdt + 1);

        var boundaryValue = 0.0;
        var boundaryCount = 0;

        // apply boundary conditions
        // xn
        if (gid.x == 0 || adjSpeeds[0] < 0) {
          boundaryValue += mur1stOrder(gid_i, 1, frac);
          boundaryCount += 1;
        }
        // xp
        if (gid.x == volSize.x - 1 || adjSpeeds[1] < 0) {
          boundaryValue += mur1stOrder(gid_i, 0, frac);
          boundaryCount += 1;
        }

        // yn
        if (gid.y == 0 || adjSpeeds[2] < 0) {
          boundaryValue += mur1stOrder(gid_i, 3, frac);
          boundaryCount += 1;
        }
        // yp
        if (gid.y == volSize.y - 1 || adjSpeeds[3] < 0) {
          boundaryValue += mur1stOrder(gid_i, 2, frac);
          boundaryCount += 1;
        }

        // zn
        if (gid.z == 0 || adjSpeeds[4] < 0) {
          boundaryValue += mur1stOrder(gid_i, 5, frac);
          boundaryCount += 1;
        }
        // zp
        if (gid.z == volSize.z - 1 || adjSpeeds[5] < 0) {
          boundaryValue += mur1stOrder(gid_i, 4, frac);
          boundaryCount += 1;
        }

        if (boundaryCount == 0) { return; }
        
        // write to the past/future texture
        textureStore(future, gid, vec4f(boundaryValue / f32(boundaryCount), 0.0, 0.0, 0.0));
      }
    `,
    label: "boundary compute module"
  });

  const boundaryComputePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: boundaryComputeModule, entryPoint: 'main' },
    label: "boundary compute pipeline"
  });

  const boundaryComputeBindGroup = (tex0, tex1) => device.createBindGroup({
    layout: boundaryComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: tex0.createView() },
      { binding: 2, resource: tex1.createView() },
      { binding: 3, resource: speedTex.createView() }
    ],
    label: "boundary compute bind group"
  });

  const renderModule = device.createShaderModule({
    code: `
      ${uniformStruct}

      @group(0) @binding(0) var<uniform> uni: Uniforms;
      @group(0) @binding(1) var stateTexture: texture_3d<f32>;
      @group(0) @binding(2) var speedTexture: texture_3d<f32>;
      @group(0) @binding(3) var stateSampler: sampler;

      struct VertexOut {
        @builtin(position) position: vec4<f32>,
        @location(0) fragCoord: vec2<f32>,
      };

      @vertex
      fn vs(@builtin(vertex_index) vIdx: u32) -> VertexOut {
        var pos = array<vec2f, 3>(
          vec2f(-1.0, -1.0),
          vec2f( 3.0, -1.0),
          vec2f(-1.0,  3.0)
        );
        var output: VertexOut;
        output.position = vec4f(pos[vIdx], 0.0, 1.0);
        output.fragCoord = 0.5 * (pos[vIdx] + vec2f(1.0)) * uni.resolution;
        return output;
      }

      // value to color: cyan -> blue -> transparent (0) -> red -> yellow
      fn transferFn(value: f32) -> vec4f {
        return clamp(vec4f(value, (abs(value) - 1) * 0.5, -value, value * value * 0.1), vec4f(0), vec4f(1,1,1,.01));
      }

      fn rayBoxIntersect(start: vec3f, dir: vec3f) -> vec2f {
        let box_min = vec3f(0);
        let box_max = uni.volSizeNorm;
        let inv_dir = 1.0 / dir;
        let tmin_tmp = (box_min - start) * inv_dir;
        let tmax_tmp = (box_max - start) * inv_dir;
        let tmin = min(tmin_tmp, tmax_tmp);
        let tmax = max(tmin_tmp, tmax_tmp);
        let t0 = max(tmin.x, max(tmin.y, tmin.z));
        let t1 = min(tmax.x, min(tmax.y, tmax.z));
        return vec2f(t0, t1);
      }

      fn pcgHash(input: f32) -> f32 {
        let state = u32(input) * 747796405u + 2891336453u;
        let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return f32((word >> 22) ^ word) / 4294967295.0;
      }

      // Convert linear color to sRGB
      fn linear2srgb(color: vec4f) -> vec4f {
        let cutoff = color.rgb < vec3f(0.0031308);
        let higher = 1.055 * pow(color.rgb, vec3f(1.0 / 2.4)) - 0.055;
        let lower = 12.92 * color.rgb;
        return vec4f(mix(higher, lower, vec3f(cutoff)), color.a);
      }

      @fragment
      fn fs(@location(0) fragCoord: vec2f) -> @location(0) vec4f {
        // Convert fragment coordinates to normalized device coordinates
        let fragNdc = fragCoord / uni.resolution * 2.0 - 1.0;

        // Project NDC to world space
        let near = uni.invMatrix * vec4f(fragNdc, 0.0, 1.0);
        let far  = uni.invMatrix * vec4f(fragNdc, 1.0, 1.0);

        // ray origin and direction
        let rayOrigin = uni.cameraPos;
        let rayDir = normalize((far.xyz / far.w) - (near.xyz / near.w));

        let intersection = rayBoxIntersect(rayOrigin, rayDir);

        // discard if ray does not intersect the box
        if (intersection.x > intersection.y || intersection.y <= 0.0) {
          // discard;
          return vec4f(0.1);
        }

        let t0 = max(intersection.x, 0.0);

        let rayDtVec = 1.0 / (uni.volSize * abs(rayDir));
        let rayDt = uni.rayDtMult * min(rayDtVec.x, min(rayDtVec.y, rayDtVec.z));

        let offset = pcgHash(fragCoord.x + uni.resolution.x * fragCoord.y) * rayDt;
        var rayPos = rayOrigin + (t0 + offset) * rayDir;

        var color = vec4f(0);

        for (var i = t0; i < intersection.y; i += rayDt) {
          let samplePos = rayPos / uni.volSizeNorm;
          rayPos += rayDir * rayDt;

          var barrier = textureSampleLevel(speedTexture, stateSampler, samplePos, 0).r;
          if (barrier <= 0.0) { // opaque barrier
            let newAlpha = (1.0 - color.a) * 0.2;
            color += vec4f(newAlpha * vec3f(0.1), newAlpha);
            break;
          }
          let sampleValue = textureSampleLevel(stateTexture, stateSampler, samplePos, 0).r;
          if (sampleValue == 0.0) { continue; } // skip empty samples

          var sampleColor = transferFn(sampleValue);

          sampleColor.a = 1.0 - pow(1.0 - sampleColor.a, uni.rayDtMult); // adjust alpha for blending
          var newAlpha = (1.0 - color.a) * sampleColor.a;
          if (samplePos.x >= 1 - 1 / uni.volSize.x) { newAlpha = 1; } // end screen

          color += vec4f(newAlpha * sampleColor.rgb, newAlpha);

          // blend sample color with accumulated color using over operator
          // if (sampleColor.a <= 0.0) { continue; } // skip fully transparent samples
          // let a = sampleColor.a * (1 - color.a);
          // color = vec4f((color.rgb * color.a + sampleColor.rgb * a) / (color.a + a), color.a + a);

          // exit if almost opaque
          if (color.a >= 0.95) { break; }
        }

        return linear2srgb(color);
      }
    `,
    label: "render module"
  });

  const filter = f32filterable ? "linear" : "nearest";
  const sampler = device.createSampler({
    magFilter: filter,
    minFilter: filter,
  });

  const renderPipeline = device.createRenderPipeline({
    label: '3d volume raycast pipeline',
    layout: 'auto',
    vertex: {
      module: renderModule,
    },
    fragment: {
      module: renderModule,
      targets: [
        {
          format: swapChainFormat,
        },
      ],
    }
  });


  const renderBindGroup = (tex) => device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: tex.createView() },
      { binding: 2, resource: speedTex.createView() },
      { binding: 3, resource: sampler },
    ],
  });

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
  const filterStrength = 10;

  const waveComputeTimingHelper = new TimingHelper(device);
  const boundaryComputeTimingHelper = new TimingHelper(device);
  const renderTimingHelper = new TimingHelper(device);

  const wgDispatchSize = [
    Math.ceil(simulationDomain[0] / wg_x),
    Math.ceil(simulationDomain[1] / wg_y),
    Math.ceil(simulationDomain[2] / wg_z)
  ]

  let [tex0, tex1] = [stateTex0, stateTex1];

  function render() {
    const startTime = performance.now();
    deltaTime += (startTime - lastFrameTime - deltaTime) / filterStrength;
    fps += (1e3 / deltaTime - fps) / filterStrength;
    lastFrameTime = startTime;

    if (keyOrbit) camera.orbit((orbleft - orbright) * KEY_ROT_SPEED, (orbup - orbdown) * KEY_ROT_SPEED);
    if (keyPan) camera.pan((panleft - panright) * KEY_PAN_SPEED, (panup - pandown) * KEY_PAN_SPEED);
    if (keyZoom) camera.zoom((zoomout - zoomin) * KEY_ZOOM_SPEED);
    if (keyFOV) camera.adjFOV((zoomout - zoomin) * KEY_FOV_SPEED);
    if (keyFOVWithoutZoom) camera.adjFOVWithoutZoom((zoomout - zoomin) * KEY_FOV_SPEED);

    const canvasTexture = context.getCurrentTexture();
    renderPassDescriptor.colorAttachments[0].view = canvasTexture.createView();

    device.queue.writeBuffer(uniformBuffer, 0, uniformValues);

    const encoder = device.createCommandEncoder();

    if (dt > 0) {
      const waveComputePass = waveComputeTimingHelper.beginComputePass(encoder);
      waveComputePass.setPipeline(waveComputePipeline);
      waveComputePass.setBindGroup(0, waveComputeBindGroup(tex0, tex1));
      waveComputePass.dispatchWorkgroups(...wgDispatchSize);
      waveComputePass.end();

      const boundaryComputePass = boundaryComputeTimingHelper.beginComputePass(encoder);
      boundaryComputePass.setPipeline(boundaryComputePipeline);
      boundaryComputePass.setBindGroup(0, boundaryComputeBindGroup(tex0, tex1));
      boundaryComputePass.dispatchWorkgroups(...wgDispatchSize);
      boundaryComputePass.end();

      [tex0, tex1] = [tex1, tex0]; // swap ping pong textures
    }

    const renderPass = renderTimingHelper.beginRenderPass(encoder, renderPassDescriptor);
    renderPass.setPipeline(renderPipeline);
    renderPass.setBindGroup(0, renderBindGroup(tex0));
    renderPass.draw(3, 1, 0, 0);
    renderPass.end();

    const commandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);

    if (dt > 0) {
      waveComputeTimingHelper.getResult().then(gpuTime => {
        waveComputeTime += (gpuTime / 1e6 - waveComputeTime) / filterStrength;
      });
      boundaryComputeTimingHelper.getResult().then(gpuTime2 => {
        boundaryComputeTime += (gpuTime2 / 1e6 - boundaryComputeTime) / filterStrength;
      });
    } else {
      waveComputeTime = boundaryComputeTime = 0;
    }
    renderTimingHelper.getResult().then(gpuTime => {
      renderTime += (gpuTime / 1e6 - renderTime) / filterStrength;
    });

    jsTime += (performance.now() - startTime - jsTime) / filterStrength;

    rafId = requestAnimationFrame(render);
  }

  intId = setInterval(() => {
    ui.fps.textContent = fps.toFixed(1);
    ui.jsTime.textContent = jsTime.toFixed(2);
    ui.frameTime.textContent = deltaTime.toFixed(1);
    ui.computeTime.textContent = (waveComputeTime + boundaryComputeTime).toFixed(3);
    ui.renderTime.textContent = renderTime.toFixed(3);
  }, 100);

  camera.updatePosition();

  uni.dtValue.set([dt]);
  uni.volSizeValue.set(simulationDomain);
  uni.volSizeNormValue.set(simulationDomainNorm);
  uni.rayDtMultValue.set([2]);
  uni.resValue.set([canvas.width, canvas.height]);

  device.queue.writeBuffer(timeBuffer, 0, new Float32Array([0]));

  rafId = requestAnimationFrame(render);
}

const camera = new Camera(defaults);


main();
