const ROT_SPEED = 0.005;
const PAN_SPEED = 0.001;
const ZOOM_SPEED = 0.0005;
const FOV_SPEED = 0.0002;

const KEY_ROT_SPEED = 3 / 15;
const KEY_PAN_SPEED = 5 / 15;
const KEY_ZOOM_SPEED = 0.01 / 15;
const KEY_FOV_SPEED = 0.005 / 15;

const minFOV = (10).toRad(), maxFOV = (120).toRad();

const defaults = {
  target: vec3.scale(simulationDomainNorm, 0.5),
  distance: 1,
  position: vec3.create(),
  azimuth: 0,
  elevation: 0,
  fov: (60).toRad(),
  near: 0.1,
  far: 100,
}

// camera state and interaction
class Camera {
  constructor(defaults) {
    this.target = vec3.clone(defaults.target);
    this.distance = defaults.distance;
    this.position = vec3.clone(defaults.position);
    this.azimuth = defaults.azimuth;
    this.elevation = defaults.elevation;

    this.fov = defaults.fov;
    this.near = defaults.near;
    this.far = defaults.far;

    this.worldUp = vec3.fromValues(0, 1, 0);
    this.updatePosition();
  }

  get viewDir() {
    return vec3.normalize(vec3.subtract(this.target, this.position));
  }

  get viewRight() {
    return vec3.normalize(vec3.cross(this.viewDir, this.worldUp));
  }

  get viewUp() {
    return vec3.normalize(vec3.cross(this.viewRight, this.viewDir));
  }


  updateMatrix() {
    const aspect = canvas.clientWidth / canvas.clientHeight;
    const proj = mat4.perspective(this.fov, aspect, this.near, this.far);
    const view = mat4.lookAt(this.position, this.target, this.worldUp);
    mat4.invert(mat4.multiply(proj, view), uni.matrixValue);

    uni.cameraPosValue.set(this.position);
  }

  updatePosition() {
    const x = Math.cos(this.elevation) * Math.sin(this.azimuth);
    const y = Math.sin(this.elevation);
    const z = Math.cos(this.elevation) * Math.cos(this.azimuth);
    this.position = vec3.scaleAndAdd(this.target, [x, y, z], this.distance);

    this.updateMatrix();

    gui.io.camFOV(this.fov.toDeg().toFixed(2));
    gui.io.camDist(this.distance.toFixed(2));
    gui.io.camTarget(vec3.toString(this.target));
    gui.io.camPos(vec3.toString(this.position));
    gui.io.camAlt(this.elevation.toDeg().toFixed(2));
    gui.io.camAz(this.azimuth.toDeg().toFixed(2));
  }

  orbit(dx, dy) {
    this.azimuth -= dx * ROT_SPEED;
    this.elevation += dy * ROT_SPEED;

    const limit = Math.PI / 2 - 0.01;
    this.elevation = this.elevation.clamp(-limit, limit);
    this.updatePosition();
  }

  pan(dx, dy, dz = 0) {
    const adjustedPanSpeed = PAN_SPEED * this.distance * this.fov;
    const pan = vec3.scaleAndAdd(
      vec3.scaleAndAdd(
        vec3.scale(this.viewRight, -dx * adjustedPanSpeed),
        this.viewUp,
        dy * adjustedPanSpeed
      ),
      this.viewDir,
      dz * adjustedPanSpeed);
    this.target = vec3.add(this.target, pan);
    this.position = vec3.add(this.position, pan);
    this.updatePosition();
  }

  zoom(delta) {
    this.distance = ((delta + 1) * this.distance).clamp(this.near, this.far);
    this.updatePosition();
  }

  adjFOV(delta) {
    this.fov = (this.fov + delta).clamp(minFOV, maxFOV);
    this.updatePosition();
  }

  adjFOVWithoutZoom(delta) {
    const initial = Math.tan(this.fov / 2) * this.distance;
    this.fov = (this.fov + delta).clamp(minFOV, maxFOV);
    this.distance = initial / Math.tan(this.fov / 2);
    this.updatePosition();
  }

  reset(e = { altKey: false, ctrlKey: false }) {
    this.fov = defaults.fov;
    if (!e.ctrlKey) this.distance = defaults.distance;
    if (!e.altKey && !e.ctrlKey) {
      this.azimuth = defaults.azimuth;
      this.elevation = defaults.elevation;
      this.target = vec3.clone(defaults.target);
    }
    this.updatePosition();
  }
}


// camera interaction state
let state = {
  orbitActive: false,
  panActive: false,
  lastX: 0,
  lastY: 0,
};

// DOM event handlers
canvas.addEventListener('contextmenu', e => e.preventDefault()); // disable context menu

canvas.addEventListener('mousedown', e => {
  if (e.button === 0) state.orbitActive = true; // left click to orbit
  if (e.button === 2) state.panActive = true;   // right click to pan
  if (e.button === 1) {
    camera.reset(e);
    camera.updatePosition();
  }
  state.lastX = e.clientX;
  state.lastY = e.clientY;
});
window.addEventListener('mouseup', e => {
  if (e.button === 0) state.orbitActive = false;
  if (e.button === 2) state.panActive = false;
});
canvas.addEventListener('mousemove', e => {
  const dx = e.clientX - state.lastX;
  const dy = e.clientY - state.lastY;
  state.lastX = e.clientX;
  state.lastY = e.clientY;

  // Orbit
  if (state.orbitActive) camera.orbit(dx, dy);

  // Pan within view-plane
  if (state.panActive) camera.pan(dx, dy);
});

canvas.addEventListener('wheel', e => {
  e.preventDefault();

  if (e.altKey) {
    // adjust FOV without zoom
    camera.adjFOVWithoutZoom(e.deltaY * FOV_SPEED)
  } else if (e.ctrlKey) {
    // FOV zoom only
    camera.adjFOV(e.deltaY * FOV_SPEED);
  } else {
    // Zoom only - move camera in/out
    camera.zoom(e.deltaY * ZOOM_SPEED);
  }
}, { passive: false });

const keyState = {
  orbit: { up: false, down: false, left: false, right: false },
  pan: { up: false, down: false, left: false, right: false, forward: false, backward: false },
  zoom: { in: false, out: false }
};

let keyOrbit = false;
let keyPan = false;
let keyZoom = false;
let keyFOV = false;
let keyFOVWithoutZoom = false;

const keyMap = {
  ArrowUp: { type: "orbit", dir: "up" },
  ArrowDown: { type: "orbit", dir: "down" },
  ArrowLeft: { type: "orbit", dir: "left" },
  ArrowRight: { type: "orbit", dir: "right" },
  w: { type: "pan", dir: "forward" },
  a: { type: "pan", dir: "left" },
  s: { type: "pan", dir: "backward" },
  d: { type: "pan", dir: "right" },
  g: { type: "pan", dir: "up" },
  v: { type: "pan", dir: "down" },
  f: { type: "zoom", dir: "in" },
  c: { type: "zoom", dir: "out" },
};

function keyCamera(e, val) {
  if (!keyMap[e.key] || e.target.tagName === "INPUT" || (e.key === "r" && e.ctrlKey)) return;

  e.preventDefault();
  const { type, dir } = keyMap[e.key];
  keyState[type][dir] = val;

  const zoomActive = keyState.zoom.in || keyState.zoom.out;

  keyOrbit = Object.values(keyState.orbit).some(Boolean);
  keyPan = Object.values(keyState.pan).some(Boolean);
  keyZoom = !(e.ctrlKey || e.altKey) && zoomActive;
  keyFOV = e.ctrlKey && zoomActive;
  keyFOVWithoutZoom = e.altKey && zoomActive;
}

window.addEventListener("keydown", (e) => {
  // console.log(e.key);
  switch (e.key) {
    case "Alt":
      e.preventDefault();
      break;
    case " ":
      e.preventDefault();
      camera.reset(e);
      break;
  }

  keyCamera(e, true);
});
window.addEventListener("keyup", (e) => {
  keyCamera(e, false);
});