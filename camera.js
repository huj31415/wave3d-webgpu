const ROT_SPEED = 0.005;
const PAN_SPEED = 0.001;
const ZOOM_SPEED = 0.0005;
const FOV_SPEED = 0.0002;

const KEY_ROT_SPEED = 3;
const KEY_PAN_SPEED = 5;
const KEY_ZOOM_SPEED = 0.01;
const KEY_FOV_SPEED = 0.005;

const minFOV = (10).toRad(), maxFOV = (120).toRad();

const defaults = {
  // target: vec3.fromValues(0.5, 0.25, 0.25),
  target: vec3.scale(simulationDomainNorm, 0.5),//vec3.fromValues(0.5, 0.5, 0.5),
  radius: 1.5,
  position: vec3.create(),
  azimuth: 0,
  elevation: 0,
  fov: (60).toRad(),
  near: 0.1,
  far: 1e5,
}

// camera state and interaction
class Camera {
  constructor(defaults) {
    this.target = vec3.clone(defaults.target);
    this.radius = defaults.radius;
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
    this.position = vec3.scaleAndAdd(this.target, [x, y, z], this.radius);

    this.updateMatrix();

    ui.camFOV.textContent = this.fov.toDeg().toFixed(2);
    ui.camDist.textContent = this.radius.toFixed(2);
    ui.camTarget.textContent = vec3.toString(this.target);
    ui.camPos.textContent = vec3.toString(this.position);
    ui.camAlt.textContent = this.elevation.toDeg().toFixed(2);
    ui.camAz.textContent = this.azimuth.toDeg().toFixed(2);
  }

  orbit(dx, dy) {
    this.azimuth -= dx * ROT_SPEED;
    this.elevation += dy * ROT_SPEED;

    const limit = Math.PI / 2 - 0.01;
    this.elevation = this.elevation.clamp(-limit, limit);
    this.updatePosition();
  }

  pan(dx, dy) {
    const adjustedPanSpeed = PAN_SPEED * this.radius * this.fov;
    const pan = vec3.scaleAndAdd(
      vec3.scale(this.viewRight, -dx * adjustedPanSpeed),
      this.viewUp,
      dy * adjustedPanSpeed
    );
    this.target = vec3.add(this.target, pan);
    this.position = vec3.add(this.position, pan);
    this.updatePosition();
  }

  zoom(delta) {
    this.radius = ((delta + 1) * this.radius).clamp(this.near, this.far);
    this.updatePosition();
  }

  adjFOV(delta) {
    this.fov = (this.fov + delta).clamp(minFOV, maxFOV);
    this.updatePosition();
  }

  adjFOVWithoutZoom(delta) {
    const initial = Math.tan(this.fov / 2) * this.radius;
    this.fov = (this.fov + delta).clamp(minFOV, maxFOV);
    this.radius = initial / Math.tan(this.fov / 2);
    this.updatePosition();
  }

  reset(e = { altKey: false, ctrlKey: false }) {
    this.fov = defaults.fov;
    if (!e.ctrlKey) this.radius = defaults.radius;
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

let orbup = false, orbdown = false, orbleft = false, orbright = false;
let panup = false, pandown = false, panleft = false, panright = false;
let zoomin = false, zoomout = false;
let keyOrbit = false, keyPan = false, keyZoom = false, keyFOV = false, keyFOVWithoutZoom = false;

function keyCamera(e, val) {
  if ((["w", "a", "s", "d", "f", "c"].includes(e.key) || e.key.includes("Arrow")) && e.target.tagName !== "INPUT") e.preventDefault();
  else return;

  switch (e.key) {
    case "ArrowUp":
      orbup = val;
      break;
    case "w":
      panup = val;
      break;
    case "ArrowDown":
      orbdown = val;
      break;
    case "s":
      pandown = val;
      break;
    case "ArrowLeft":
      orbleft = val;
      break;
    case "a":
      panleft = val;
      break;
    case "ArrowRight":
      orbright = val;
      break;
    case "d":
      panright = val;
      break;
    case "f":
      zoomin = val;
      break;
    case "c":
      zoomout = val;
      break;
  }

  const zoom = zoomin || zoomout;

  keyOrbit = orbup || orbdown || orbleft || orbright;
  keyPan = panup || pandown || panleft || panright;
  keyZoom = !(e.ctrlKey || e.altKey) && zoom;
  keyFOV = e.ctrlKey && zoom;
  keyFOVWithoutZoom = e.altKey && zoom;
}
window.addEventListener("keydown", (e) => {
  console.log(e.key);
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