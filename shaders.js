/* 27 point stencil

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
*/

const waveShaderCode = (wg_x, wg_y, wg_z) => /* wgsl */`
${uni.uniformStruct}

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var past_future:  texture_storage_3d<r32float, read_write>;
@group(0) @binding(2) var present:      texture_storage_3d<r32float, read>;
@group(0) @binding(3) var waveSpeed:    texture_storage_3d<r32float, read>;
@group(0) @binding(4) var energy:    texture_storage_3d<r32float, read_write>;

const WG_X: u32 = ${wg_x};
const WG_Y: u32 = ${wg_y};
const WG_Z: u32 = ${wg_z};

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
    textureStore(energy, gid, vec4f(0));
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

  // write to energy texture (adds several ms to compute time)
  if (uni.energyFilter > 0) {
    let current = textureLoad(energy, gid);
    textureStore(energy, gid, current + (newValue * newValue - current) / uni.energyFilter);
  }
}
`;


const boundaryShaderCode = (wg_x, wg_y, wg_z) => /* wgsl */`
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
`;


const renderShaderCode = /* wgsl */`
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
//   //  color *= select(1.0, 5.0, renderEnergy && samplePos.x >= 1 - uni.rayDtMult / uni.volSize.x);
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
  let renderEnergy = uni.energyFilter > 0;
  let energyMult = select(1.0, uni.energyMult, renderEnergy);
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
    let sampleValue = textureSampleLevel(stateTexture, stateSampler, samplePos, 0).r * energyMult;

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
    if (renderEnergy && samplePos.x >= plusXLimit) {
      sampleColor.a *= uni.plusXAlpha;
    }
    // sampleColor.a *= 1 + f32(renderEnergy && samplePos.x >= plusXLimit) * (uni.plusXAlpha - 1);

    // Exponential blending
    color += (1.0 - color.a) * (1.0 - exp(-sampleColor.a * adjDt)) * vec4f(sampleColor.rgb, 1);

    // Exit if nearly opaque
    // if (color.a >= 0.95) {
    //   break;
    // }
  }
  return linear2srgb(color);
}
`