
/*
 * vec3 and mat4 utility classes
 * Provides basic vector and matrix operations for 3D graphics
 */

class vec3 {
  // create a zero vector
  static create() {
    return new Float32Array(3);
  }

  // create a vector from values
  static fromValues(x, y, z) {
    const out = new Float32Array(3);
    out[0] = x;
    out[1] = y;
    out[2] = z;
    return out;
  }

  // out = a + b
  static add(a, b, out = this.create()) {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
    return out;
  }

  // out = a - b
  static subtract(a, b, out = this.create()) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
    return out;
  }

  // out = a * scalar
  static scale(a, scalar, out = this.create()) {
    out[0] = a[0] * scalar;
    out[1] = a[1] * scalar;
    out[2] = a[2] * scalar;
    return out;
  }

  // out = a + (b * scalar)
  static scaleAndAdd(a, b, scalar, out = this.create()) {
    out[0] = a[0] + b[0] * scalar;
    out[1] = a[1] + b[1] * scalar;
    out[2] = a[2] + b[2] * scalar;
    return out;
  }

  // compute dot product of a and b
  static dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  // compute cross product of a and b, store in out
  static cross(a, b, out = this.create()) {
    const ax = a[0], ay = a[1], az = a[2];
    const bx = b[0], by = b[1], bz = b[2];
    out[0] = ay * bz - az * by;
    out[1] = az * bx - ax * bz;
    out[2] = ax * by - ay * bx;
    return out;
  }

  static length(a) {
    return Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
  }

  // normalize vector a, store in out
  static normalize(a, out = this.create()) {
    let len = this.length(a);
    if (len > 0) {
      out = this.scale(a, 1 / len);
    }
    return out;
  }

  static clone(a, out = this.create()) {
    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
    return out;
  }

  static toString(a) {
    return Array.from(a).map((i) => parseFloat(i).toFixed(2));
  }

  static equals(a, b) {
    return a.every((e, i) => e === b[i]);
  }
}

class mat4 {
  // create identity matrix
  static create() {
    const out = new Float32Array(16);
    out[0] = 1;
    out[5] = 1;
    out[10] = 1;
    out[15] = 1;
    return out;
  }

  // generate perspective projection matrix
  static perspective(fovy, aspect, near, far, out = this.create()) {
    const f = 1.0 / Math.tan(fovy / 2);
    const nf = 1 / (near - far);
    out[0] = f / aspect;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;

    out[4] = 0;
    out[5] = f;
    out[6] = 0;
    out[7] = 0;

    out[8] = 0;
    out[9] = 0;
    out[10] = (far + near) * nf;
    out[11] = -1;

    out[12] = 0;
    out[13] = 0;
    out[14] = (2 * far * near) * nf;
    out[15] = 0;
    return out;
  }

  // generate lookAt view matrix
  static lookAt(eye, center, up, out = this.create()) {
    // z along view direction
    let z = vec3.normalize(vec3.subtract(eye, center));

    // x = ||up x z||
    let x = vec3.normalize(vec3.cross(up, z));

    // y = ||z x x||
    let y = vec3.cross(z, x);

    out[0] = x[0];
    out[1] = y[0];
    out[2] = z[0];
    out[3] = 0;

    out[4] = x[1];
    out[5] = y[1];
    out[6] = z[1];
    out[7] = 0;

    out[8] = x[2];
    out[9] = y[2];
    out[10] = z[2];
    out[11] = 0;

    out[12] = -vec3.dot(x, eye);
    out[13] = -vec3.dot(y, eye);
    out[14] = -vec3.dot(z, eye);
    out[15] = 1;

    return out;
  }

  // multiply two 4x4 matrices: out = a * b
  static multiply(a, b, out = this.create()) {
    for (let i = 0; i < 4; i++) {
      const ai0 = a[i], ai1 = a[i + 4], ai2 = a[i + 8], ai3 = a[i + 12];
      out[i] = ai0 * b[0] + ai1 * b[1] + ai2 * b[2] + ai3 * b[3];
      out[i + 4] = ai0 * b[4] + ai1 * b[5] + ai2 * b[6] + ai3 * b[7];
      out[i + 8] = ai0 * b[8] + ai1 * b[9] + ai2 * b[10] + ai3 * b[11];
      out[i + 12] = ai0 * b[12] + ai1 * b[13] + ai2 * b[14] + ai3 * b[15];
    }
    return out;
  }

  // from gl-matrix
  static invert(a, out = this.create()) {
    let a00 = a[0],
      a01 = a[1],
      a02 = a[2],
      a03 = a[3],
      a10 = a[4],
      a11 = a[5],
      a12 = a[6],
      a13 = a[7],
      a20 = a[8],
      a21 = a[9],
      a22 = a[10],
      a23 = a[11],
      a30 = a[12],
      a31 = a[13],
      a32 = a[14],
      a33 = a[15];

    let b00 = a00 * a11 - a01 * a10;
    let b01 = a00 * a12 - a02 * a10;
    let b02 = a00 * a13 - a03 * a10;
    let b03 = a01 * a12 - a02 * a11;
    let b04 = a01 * a13 - a03 * a11;
    let b05 = a02 * a13 - a03 * a12;
    let b06 = a20 * a31 - a21 * a30;
    let b07 = a20 * a32 - a22 * a30;
    let b08 = a20 * a33 - a23 * a30;
    let b09 = a21 * a32 - a22 * a31;
    let b10 = a21 * a33 - a23 * a31;
    let b11 = a22 * a33 - a23 * a32;

    // Calculate the determinant
    let det =
      b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;

    if (!det) {
      return null;
    }
    det = 1.0 / det;

    out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
    out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
    out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
    out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
    out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
    out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
    out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
    out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
    out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
    out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
    out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
    out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
    out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
    out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
    out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
    out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;

    return out;
  }

  // rotate matrix a by rad around X axis, store in out
  static rotateX(rad, a, out = this.create()) {
    const s = Math.sin(rad);
    const c = Math.cos(rad);
    // copy a into out
    out.set(a);

    // rows 1 and 2 transform
    out[4] = a[4] * c + a[8] * s;
    out[5] = a[5] * c + a[9] * s;
    out[6] = a[6] * c + a[10] * s;
    out[7] = a[7] * c + a[11] * s;

    out[8] = a[8] * c - a[4] * s;
    out[9] = a[9] * c - a[5] * s;
    out[10] = a[10] * c - a[6] * s;
    out[11] = a[11] * c - a[7] * s;
    return out;
  }

  // rotate matrix a by rad around Y axis, store in out
  static rotateY(rad, a, out = this.create()) {
    const s = Math.sin(rad);
    const c = Math.cos(rad);
    out.set(a);

    // rows 0 and 2 transform
    out[0] = a[0] * c - a[8] * s;
    out[1] = a[1] * c - a[9] * s;
    out[2] = a[2] * c - a[10] * s;
    out[3] = a[3] * c - a[11] * s;

    out[8] = a[0] * s + a[8] * c;
    out[9] = a[1] * s + a[9] * c;
    out[10] = a[2] * s + a[10] * c;
    out[11] = a[3] * s + a[11] * c;
    return out;
  }
}