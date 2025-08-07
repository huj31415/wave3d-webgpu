// Supports WGSL types f32, vec2f, vec3f, vec4f, mat4x4f.

class Uniforms {
  constructor() {
    this.layout = [];
    this.byteOffset = 0;
    this.byteLength = 0;
    this.gpuBuffer = null;
    this.uniformData = null;
    this.values = {}; // Named views for direct access
    this.uniformStruct = null;
  }

  // Add a new uniform field
  addUniform(name, type) {
    // check if name is already used?
    const info = Uniforms.typeInfo[type];
    if (!info) throw new Error(`Unsupported type: ${type}`);
    // align current offset
    this.byteOffset = Math.ceil(this.byteOffset / info.align) * info.align;
    this.layout.push({ name, type, offset: this.byteOffset, size: info.size });
    this.byteOffset += info.size;
  }

  // Finalize layout and generate WGSL struct
  finalize() {
    // Round total size up to 16 bytes
    this.byteLength = Math.ceil(this.byteOffset / 16) * 16;
    // CPU-side storage
    this.uniformData = new Float32Array(this.byteLength / 4);

    // Create named subarray views for each uniform
    for (const { name, type, offset, size } of this.layout) {
      const floatOffset = offset / 4;
      const floatLength = size / 4;
      this.values[name] = this.uniformData.subarray(floatOffset, floatOffset + floatLength);
    }

    // Build WGSL struct
    const lines = ["struct Uniforms {"];
    for (const { name, type, offset, size } of this.layout) {
      lines.push(`  ${name}: ${type},`);
    }
    lines.push("};");
    this.uniformStruct = lines.join("\n");
  }

  // Set uniform value (array or number)
  set(name, value) {
    if (!(name in this.values)) throw new Error(`Uniform not found: ${name}`);
    const target = this.values[name];
    const source = (typeof value === "number")
      ? [value]
      : value instanceof Float32Array
        ? value
        : new Float32Array(value);
    target.set(source);
  }

  // Allocate buffer on GPU, overwriting if it already exists, and returns the new buffer
  createBuffer(device) {
    if (this.gpuBuffer) {
      console.log("Buffer already exists, overwriting");
      this.gpuBuffer.destroy();
    }
    this.gpuBuffer = device.createBuffer({
      size: this.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    return this.gpuBuffer;
  }

  // Upload to GPU
  update(queue) {
    queue.writeBuffer(this.gpuBuffer, 0, this.uniformData.buffer, 0, this.byteLength);
  }
}

Uniforms.typeInfo = {
  "f32":     { size: 4,  align: 4 },
  "vec2f":   { size: 8,  align: 8 },
  "vec3f":   { size: 12, align: 16 },
  "vec4f":   { size: 16, align: 16 },
  "mat4x4f": { size: 16*4, align: 16 },
  "vec2<f32>":   { size: 8,  align: 8 },
  "vec3<f32>":   { size: 12, align: 16 },
  "vec4<f32>":   { size: 16, align: 16 },
  "mat4x4<f32>": { size: 16*4, align: 16 },
};
