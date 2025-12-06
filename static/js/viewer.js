// viewer.js (ES Module)

// Imports — substituem os antigos scripts deprecated
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/GLTFLoader.js";

// Socket.IO ESModule-compatible dynamic import
import io from "https://cdn.jsdelivr.net/npm/socket.io-client/dist/socket.io.esm.min.js";

// DOM
const container = document.getElementById("viewer");
const width = container.clientWidth || window.innerWidth * 0.8;
const height = container.clientHeight || window.innerHeight * 0.6;

// Three.js setup
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
camera.position.set(0, 1.6, 3);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(width, height);
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1.2, 0);
controls.update();

const light = new THREE.HemisphereLight(0xffffff, 0x444444);
light.position.set(0, 1, 0);
scene.add(light);

// Animation system
let mixer = null;
let actions = {};
let activeAction = null;

// Load GLB
const loader = new GLTFLoader();
loader.load(
    "/models/avatar.glb",
    (gltf) => {
        scene.add(gltf.scene);

        if (gltf.animations && gltf.animations.length > 0) {
            mixer = new THREE.AnimationMixer(gltf.scene);

            const anims = gltf.animations;

            // Buscar por nomes
            let knownAnim = anims.find(a => a.name?.toLowerCase().includes("known"));
            let unknownAnim = anims.find(a => a.name?.toLowerCase().includes("unknown"));

            // Fallback por índice
            if (!knownAnim && anims.length > 1) knownAnim = anims[1];
            if (!unknownAnim && anims.length > 0) unknownAnim = anims[0];

            if (unknownAnim) actions["unknown"] = mixer.clipAction(unknownAnim);
            if (knownAnim) actions["known"] = mixer.clipAction(knownAnim);

            // Animação inicial
            if (actions["unknown"]) {
                activeAction = actions["unknown"];
                activeAction.play();
            }
        }
    },
    undefined,
    (err) => console.error("Erro carregando GLB:", err)
);

// Animation loop
const clock = new THREE.Clock();
function animate() {
    requestAnimationFrame(animate);
    if (mixer) mixer.update(clock.getDelta());
    renderer.render(scene, camera);
}
animate();

// Troca de animação com crossfade
function fadeTo(name, duration = 0.3) {
    if (!mixer || !actions[name]) return;
    const toAction = actions[name];
    if (activeAction === toAction) return;

    toAction.reset().play();
    toAction.crossFadeFrom(activeAction, duration, true);
    activeAction = toAction;
}

// Socket.IO
const socket = io();

socket.on("connect", () => {
    console.log("Conectado ao servidor");
    socket.emit("start_rec", {});
});

socket.on("recognition", (data) => {
    console.log("Reconhecimento:", data);
    if (data.status === "known") fadeTo("known", 0.5);
    else fadeTo("unknown", 0.5);
});

// Responsividade
window.addEventListener("resize", () => {
    const w = container.clientWidth;
    const h = container.clientHeight;

    camera.aspect = w / h;
    camera.updateProjectionMatrix();

    renderer.setSize(w, h);
});
