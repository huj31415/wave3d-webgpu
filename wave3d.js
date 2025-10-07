let adapter, device;
let gpuInfo = false;

async function main() {

  if (device) device.destroy();

  let maxComputeInvocationsPerWorkgroup, maxBufferSize, f32filterable;

  // WebGPU Setup
  // if (!device) {
  adapter = await navigator.gpu?.requestAdapter();

  maxComputeInvocationsPerWorkgroup = adapter.limits.maxComputeInvocationsPerWorkgroup;
  maxBufferSize = adapter.limits.maxBufferSize;
  f32filterable = adapter.features.has("float32-filterable");

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
    const msg = event.error.message;
    if (msg.includes("max buffer size limit"))
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

  textures.stateTex0 = device.createTexture({
    size: simulationDomain,
    dimension: "3d",
    format: "r32float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
    label: "state texture 0",
  });
  textures.stateTex1 = device.createTexture({
    size: simulationDomain,
    dimension: "3d",
    format: "r32float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
    label: "state texture 1",
  });
  textures.intensityTex = device.createTexture({
    size: simulationDomain,
    dimension: "3d",
    format: "r32float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
    label: "intensity texture",
  });
  textures.speedTex = device.createTexture({
    size: simulationDomain,
    dimension: "3d",
    format: "r32float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
    label: "wavespeed texture",
  });
  updateSpeedTexture();

  const uniformBuffer = uni.createBuffer(device);


  const waveComputeModule = device.createShaderModule({
    code: /* wgsl */`
      ${uni.uniformStruct}

      @group(0) @binding(0) var<uniform> uni: Uniforms;
      @group(0) @binding(1) var past_future:  texture_storage_3d<r32float, read_write>;
      @group(0) @binding(2) var present:      texture_storage_3d<r32float, read>;
      @group(0) @binding(3) var waveSpeed:    texture_storage_3d<r32float, read>;
      @group(0) @binding(4) var intensity:    texture_storage_3d<r32float, read_write>;

      const WG_X: u32 = ${wg_x};
      const WG_Y: u32 = ${wg_y};
      const WG_Z: u32 = ${wg_z};

      // const directions: array<vec3i, 26> = array<vec3i, 26>(
      //   // 00-05 orthogonal directions (cubic faces)
      //   vec3i(-1,  0,  0), // xn
      //   vec3i( 1,  0,  0), // xp
      //   vec3i( 0, -1,  0), // yn
      //   vec3i( 0,  1,  0), // yp
      //   vec3i( 0,  0, -1), // zn
      //   vec3i( 0,  0,  1), // zp

      //   // 06-17 planar diagonals (cubic edges)
      //   vec3i(-1, -1,  0), // xnyn
      //   vec3i(-1,  1,  0), // xnyp
      //   vec3i( 1, -1,  0), // xpyn
      //   vec3i( 1,  1,  0), // xpyp

      //   vec3i(-1,  0, -1), // xnzn
      //   vec3i(-1,  0,  1), // xnzp
      //   vec3i( 1,  0, -1), // xpzn
      //   vec3i( 1,  0,  1), // xpzp

      //   vec3i( 0, -1, -1), // ynzn
      //   vec3i( 0, -1,  1), // ynzp
      //   vec3i( 0,  1, -1), // ypzn
      //   vec3i( 0,  1,  1), // ypzp

      //   // 18-25 nonplanar diagonals (cubic corners)
      //   vec3i(-1, -1, -1), // xnynzn
      //   vec3i(-1, -1,  1), // xnynzp
      //   vec3i(-1,  1, -1), // xnypzn
      //   vec3i(-1,  1,  1), // xnypzp
      //   vec3i( 1, -1, -1), // xpynzn
      //   vec3i( 1, -1,  1), // xpynzp
      //   vec3i( 1,  1, -1), // xpypzn
      //   vec3i( 1,  1,  1), // xpypzp
      // );

      const directions: array<vec3i, 6> = array<vec3i, 6>(
        // 00-05 orthogonal directions (cubic faces)
        vec3i(-1,  0,  0), // xn
        vec3i( 1,  0,  0), // xp
        vec3i( 0, -1,  0), // yn
        vec3i( 0,  1,  0), // yp
        vec3i( 0,  0, -1), // zn
        vec3i( 0,  0,  1), // zp
      );


      // var<workgroup> tile: array<f32, (WG_X + 2) * (WG_Y + 2) * (WG_Z + 2)>;

      // fn tileIndex(idx: vec3i) -> u32 {
      //   let sidx = vec3u(idx + vec3i(1)); // shift by 1 to account for halo
      //   return sidx.x + (WG_X + 2u) * (sidx.y + (WG_Y + 2u) * sidx.z);
      // }

      // 3d wave compute shader
      @compute @workgroup_size(WG_X, WG_Y, WG_Z)
      fn main(
        @builtin(global_invocation_id) gid: vec3u,
        @builtin(local_invocation_id) lid: vec3u
      ) {
        // var earlyReturn = false;

        let volSize = vec3u(uni.volSize);
        let gid_i = vec3i(gid);
        // let lid_i = vec3i(lid);

        // check if the index is within bounds
        if (any(gid >= volSize)) { return; }

        // read the wave speed and write 0 if barrier
        let cdt = textureLoad(waveSpeed, gid).r * uni.dt;
        if (cdt <= 0) {
          textureStore(past_future, gid, vec4f(0));
          textureStore(intensity, gid, vec4f(0));
          return;
          // earlyReturn = true;
        }

        // wave
        if (uni.waveOn > 0) {
          // write wave source to texture
          if ((uni.waveSourceType == 0 && gid.x == 0)
            || (uni.waveSourceType == 1 && gid.x == 0 && all(abs(gid_i.yz - vec2i(uni.wavePos.yz)) <= vec2i(uni.waveHalfSize)))
            || (uni.waveSourceType == 2 && all(gid == vec3u(uni.wavePos)))
          ) {
            // write to the past/future texture
            textureStore(past_future, gid, vec4f(uni.waveValue, 0.0, 0.0, 0.0) * select(200.0, 1.0, uni.waveSourceType <= 1));
            return;
            // earlyReturn = true;
          }
        }

        // read the states
        let pastValue = textureLoad(past_future, gid).r;
        let presentValue = textureLoad(present, gid).r;

        var laplacian = -6.0 * presentValue
          + textureLoad(present, gid_i + directions[0]).r
          + textureLoad(present, gid_i + directions[1]).r
          + textureLoad(present, gid_i + directions[2]).r
          + textureLoad(present, gid_i + directions[3]).r
          + textureLoad(present, gid_i + directions[4]).r
          + textureLoad(present, gid_i + directions[5]).r;

        // tile[tileIndex(lid_i)] = presentValue;
        // // load halo
        // if (lid.x == 0 || lid.x == WG_X - 1u) {
        //   let dir = select(directions[1], directions[0], lid.x == 0);
        //   tile[tileIndex(lid_i + dir)] = textureLoad(present, gid_i + dir).r;
        // //   tile[tileIndex(lid_i + directions[0])] = textureLoad(present, gid_i + directions[0]).r;
        // // } else if (lid.x == WG_X - 1u) {
        // //   tile[tileIndex(lid_i + directions[1])] = textureLoad(present, gid_i + directions[1]).r;
        // }

        // if (lid.y == 0 || lid.y == WG_Y - 1u) {
        //   let dir = select(directions[3], directions[2], lid.y == 0);
        //   tile[tileIndex(lid_i + dir)] = textureLoad(present, gid_i + dir).r;
        // //   tile[tileIndex(lid_i + directions[2])] = textureLoad(present, gid_i + directions[2]).r;
        // // } else if (lid.y == WG_Y - 1u) {
        // //   tile[tileIndex(lid_i + directions[3])] = textureLoad(present, gid_i + directions[3]).r;
        // }

        // if (lid.z == 0 || lid.z == WG_Z - 1u) {
        //   let dir = select(directions[5], directions[4], lid.z == 0);
        //   tile[tileIndex(lid_i + dir)] = textureLoad(present, gid_i + dir).r;
        // //   tile[tileIndex(lid_i + directions[4])] = textureLoad(present, gid_i + directions[4]).r;
        // // } else if (lid.z == WG_Z - 1u) {
        // //   tile[tileIndex(lid_i + directions[5])] = textureLoad(present, gid_i + directions[5]).r;
        // }

        // workgroupBarrier();

        // if (earlyReturn) { return; };

        // let laplacian = -6.0 * presentValue
        //    + tile[tileIndex(lid_i + directions[0])]
        //    + tile[tileIndex(lid_i + directions[1])]
        //    + tile[tileIndex(lid_i + directions[2])]
        //    + tile[tileIndex(lid_i + directions[3])]
        //    + tile[tileIndex(lid_i + directions[4])]
        //    + tile[tileIndex(lid_i + directions[5])];

        // 26 point stencil
        // var laplacian = -88.0 * presentValue;

        // for (var i = 0; i < 6; i++) {
        //   laplacian += 6 * textureLoad(present, gid_i + directions[i]).r;
        // }
        // for (var i = 6; i < 18; i++) {
        //   laplacian += 3 * textureLoad(present, gid_i + directions[i]).r;
        // }
        // for (var i = 18; i < 26; i++) {
        //   laplacian += 2 * textureLoad(present, gid_i + directions[i]).r;
        // }

        // laplacian /= 26;

        // compute the new value based on the wave equation
        let newValue = 2 * presentValue - pastValue + cdt * cdt * laplacian;

        // write to the past/future texture
        textureStore(past_future, gid, vec4f(newValue, 0.0, 0.0, 0.0));

        // write to intensity texture (adds several ms to compute time)
        if (uni.intensityFilter > 0) {
          let current = textureLoad(intensity, gid);
          textureStore(intensity, gid, current + (newValue * newValue - current) / uni.intensityFilter);
        }
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
      { binding: 3, resource: textures.speedTex.createView() },
      { binding: 4, resource: textures.intensityTex.createView() },
    ],
    label: "wave compute bind group"
  });

  const boundaryComputeModule = device.createShaderModule({
    code: /* wgsl */`
      ${uni.uniformStruct}

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

      // frac1 = (cdt - 1) / (cdt + 1);
      // frac2 = dt * cdt / (2 * (1 + cdt))
      fn mur2ndOrder(gid_i: vec3i, index: u32, frac1: f32, frac2: f32, iAdj1: u32, iAdj2: u32) -> f32 {
        let idx = gid_i + directions[index];
        let presentValue = textureLoad(present, idx).r;
        return presentValue + frac1 * (textureLoad(future, idx).r - textureLoad(future, gid_i).r)
          + frac2 * (
            textureLoad(present, gid_i + directions[iAdj1]).r - 2 * presentValue + textureLoad(present, gid_i + directions[iAdj1 + 1]).r
            + textureLoad(present, gid_i + directions[iAdj2]).r - 2 * presentValue + textureLoad(present, gid_i + directions[iAdj2 + 1]).r
          );
      }

      // 3d wave boundary compute shader
      @compute @workgroup_size(WG_X, WG_Y, WG_Z)
      fn main(
        @builtin(global_invocation_id) gid: vec3u
      ) {

        let volSize = vec3u(uni.volSize);
        let gid_i = vec3i(gid);

        // check if the index is within bounds
        if (any(gid >= volSize)) { return; }

        // read the wave speed
        let cdt = textureLoad(waveSpeed, gid).r * uni.dt;
        if (cdt <= 0) { return; }

        let adjSpeeds = array<f32, 6>(
          textureLoad(waveSpeed, gid_i + directions[0]).r, // left
          textureLoad(waveSpeed, gid_i + directions[1]).r, // right
          textureLoad(waveSpeed, gid_i + directions[2]).r, // down
          textureLoad(waveSpeed, gid_i + directions[3]).r, // up
          textureLoad(waveSpeed, gid_i + directions[4]).r, // back
          textureLoad(waveSpeed, gid_i + directions[5]).r  // front
        );
        let frac = (cdt - 1) / (cdt + 1);
        let frac2 = uni.dt * cdt / (2 * (1 + cdt));

        var boundaryValue = 0.0;
        var boundaryCount = 0;

        // apply boundary conditions
        // xn
        if ((uni.waveOn == 0 && gid.x == 0) || adjSpeeds[0] < 0) {
          // boundaryValue += mur1stOrder(gid_i, 1, frac);
          boundaryValue += mur2ndOrder(gid_i, 1, frac, frac2, 2, 4);
          boundaryCount += 1;
        }
        // xp
        if (gid.x == volSize.x - 1 || adjSpeeds[1] < 0) {
          // boundaryValue += mur1stOrder(gid_i, 0, frac);
          boundaryValue += mur2ndOrder(gid_i, 0, frac, frac2, 2, 4);
          boundaryCount += 1;
        }

        // yn
        if (gid.y == 0 || adjSpeeds[2] < 0) {
          // boundaryValue += mur1stOrder(gid_i, 3, frac);
          boundaryValue += mur2ndOrder(gid_i, 3, frac, frac2, 0, 4);
          boundaryCount += 1;
        }
        // yp
        if (gid.y == volSize.y - 1 || adjSpeeds[3] < 0) {
          // boundaryValue += mur1stOrder(gid_i, 2, frac);
          boundaryValue += mur2ndOrder(gid_i, 2, frac, frac2, 0, 4);
          boundaryCount += 1;
        }

        // zn
        if (gid.z == 0 || adjSpeeds[4] < 0) {
          // boundaryValue += mur1stOrder(gid_i, 5, frac);
          boundaryValue += mur2ndOrder(gid_i, 5, frac, frac2, 0, 2);
          boundaryCount += 1;
        }
        // zp
        if (gid.z == volSize.z - 1 || adjSpeeds[5] < 0) {
          // boundaryValue += mur1stOrder(gid_i, 4, frac);
          boundaryValue += mur2ndOrder(gid_i, 4, frac, frac2, 0, 2);
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
      { binding: 3, resource: textures.speedTex.createView() }
    ],
    label: "boundary compute bind group"
  });

  const renderModule = device.createShaderModule({
    code: /* wgsl */`
      ${uni.uniformStruct}

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

      // // value to color: cyan -> blue -> transparent (0) -> red -> yellow
      // fn transferFn(value: f32) -> vec4f {
      //   // let a = 1.0 - pow(1.0 - clamp(value * value * 0.1, 0, 0.01), uni.rayDtMult);
      //   let a = clamp(value * value * 0.1, 0, 0.01) * uni.globalAlpha;
      //   return clamp(vec4f(value, (abs(value) - 1) * 0.5, -value, a), vec4f(0), vec4f(1)) * 10; // 10x for beer-lambert
      // }

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
        return vec4f(select(higher, lower, cutoff), color.a);
      }

      // // Old alpha blending method, add to accumulated color
      // fn oldBlend(sampleColor: vec4f, accumulatedColor: vec4f) -> vec4f {
      //   return vec4f(sampleColor.rgb, 1) * (1.0 - accumulatedColor.a) * sampleColor.a;
      //   //  color *= select(1.0, 5.0, renderIntensity && samplePos.x >= 1 - uni.rayDtMult / uni.volSize.x);
      // }

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
        var rayPos = (rayOrigin + (t0 + offset) * rayDir) / uni.volSizeNorm;
        let rayDirNorm = rayDir / uni.volSizeNorm;

        var color = vec4f(0);
        let renderIntensity = uni.intensityFilter > 0;
        let intensityMult = select(1.0, uni.intensityMult, renderIntensity);
        let plusXLimit = 1.0 - uni.rayDtMult / uni.volSize.x;

        var remainingDist = intersection.y - t0;

        loop {
          if (remainingDist <= 0.0) { break; }

          let adjDt = min(rayDt, remainingDist);
          let samplePos = rayPos;

          // Precompute position increment
          rayPos += rayDirNorm * adjDt;
          remainingDist -= adjDt;

          // Sample speed
          let speed = textureSampleLevel(speedTexture, stateSampler, samplePos, 0).r;

          // Early exit on barrier
          if (speed <= 0.0) {
            // Barrier blend
            color += vec4f((1.0 - color.a) * (1.0 - exp(-adjDt)));; // Barrier color
            break;
          }

          // Sample state
          let sampleValue = textureSampleLevel(stateTexture, stateSampler, samplePos, 0).r * intensityMult;

          // Skip if empty and not a boundary
          if (sampleValue == 0.0 && speed == 1.0) {
            continue;
          }
          
          // inlined transfer function
          let a = clamp(sampleValue * sampleValue * 0.1, 0, 0.01) * uni.globalAlpha;
          var sampleColor = clamp(vec4f(sampleValue, (abs(sampleValue) - 1) * 0.5, -sampleValue, a), vec4f(0), vec4f(1)) * 10;

          // Slight opacity to refractive materials
          sampleColor += f32(speed != 1.0) * vec4f(min(abs(1.0 - speed), 0.05) * 6);

          // +X face emphasis
          if (renderIntensity && samplePos.x >= plusXLimit) {
            sampleColor.a *= uni.plusXAlpha;
          }
          // sampleColor.a *= 1 + f32(renderIntensity && samplePos.x >= plusXLimit) * (uni.plusXAlpha - 1);

          // Exponential blending
          color += (1.0 - color.a) * (1.0 - exp(-sampleColor.a * adjDt)) * vec4f(sampleColor.rgb, 1);

          // Exit if nearly opaque
          // if (color.a >= 0.95) {
          //   break;
          // }
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
      { binding: 2, resource: textures.speedTex.createView() },
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
  const filterStrength = 50;

  const waveComputeTimingHelper = new TimingHelper(device);
  const boundaryComputeTimingHelper = new TimingHelper(device);
  const renderTimingHelper = new TimingHelper(device);

  const wgDispatchSize = [
    Math.ceil(simulationDomain[0] / wg_x),
    Math.ceil(simulationDomain[1] / wg_y),
    Math.ceil(simulationDomain[2] / wg_z)
  ]

  let [tex0, tex1] = [textures.stateTex0, textures.stateTex1];

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

    const run = dt > 0

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
    renderPass.setBindGroup(0, renderBindGroup(intensityFilterStrength > 0 ? textures.intensityTex : tex0));
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
  uni.values.intensityFilter.set([intensityFilterStrength]);
  uni.values.intensityMult.set([1]);
  uni.values.waveSourceType.set([0]);
  uni.values.globalAlpha.set([2]);
  uni.values.plusXAlpha.set([2]);
  uni.values.wavePos.set(wavePos);
  uni.values.waveHalfSize.set(waveHalfSize);
  time = 0;

  rafId = requestAnimationFrame(render);
}

const camera = new Camera(defaults);

main().then(() => quadSymmetricFlatPreset(flatPresets.ZonePlate, presetXOffset, presetThickness, { shape: shapes.circular, f: 192, nCutouts: 4 }));