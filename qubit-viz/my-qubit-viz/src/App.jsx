import React, { useState, useRef, useCallback, useEffect, useMemo } from 'react';
// Requires THREE.js, OrbitControls, CSS2DRenderer loaded externally

/**
 * ComplexPlanePlot Component (Unchanged)
 */
const ComplexPlanePlot = ({
  x, y, onDrag, maxRadius = Infinity,
  xLabel, yLabel, title, pointId,
  bgColor = '#f0f0f0', axisColor = '#999', pointColor = 'royalblue',
  pathPoints = [], pathColor = 'rgba(0, 0, 0, 0.3)',
}) => {
  const svgRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const [viewBox] = useState({ minX: -1.5, minY: -1.5, width: 3, height: 3 });

  const getSVGCoordinates = useCallback((event) => {
    if (!svgRef.current) return null;
    const CTM = svgRef.current.getScreenCTM();
    if (!CTM) return null;
    const clientX = event.clientX ?? event.touches[0].clientX;
    const clientY = event.clientY ?? event.touches[0].clientY;
    let point = svgRef.current.createSVGPoint();
    point.x = clientX;
    point.y = clientY;
    point = point.matrixTransform(CTM.inverse());
    return { x: point.x, y: point.y };
  }, [svgRef]);

  const handleMouseDown = useCallback((event) => {
    if (event.target.id === `${pointId}-handle`) {
      event.preventDefault();
      setIsDragging(true);
    }
  }, [pointId]);

   const handleMouseMove = useCallback((event) => {
    if (isDragging && svgRef.current) {
      event.preventDefault();
      const coords = getSVGCoordinates(event);
      if (coords) {
        let newX = coords.x;
        let newY = coords.y;
        newX = Math.max(viewBox.minX + 0.01, Math.min(viewBox.minX + viewBox.width - 0.01, newX));
        newY = Math.max(viewBox.minY + 0.01, Math.min(viewBox.minY + viewBox.height - 0.01, newY));
        onDrag(newX, newY); // Pass raw dragged coordinates
      }
    }
  }, [isDragging, getSVGCoordinates, onDrag, viewBox]);

  const handleMouseUp = useCallback((event) => {
    if (isDragging) {
      event.preventDefault();
      setIsDragging(false);
    }
  }, [isDragging]);

  useEffect(() => {
    const moveHandler = (event) => handleMouseMove(event);
    const upHandler = (event) => handleMouseUp(event);
    if (isDragging) {
      window.addEventListener('mousemove', moveHandler);
      window.addEventListener('mouseup', upHandler);
      window.addEventListener('touchmove', moveHandler, { passive: false });
      window.addEventListener('touchend', upHandler);
    }
    return () => {
      window.removeEventListener('mousemove', moveHandler);
      window.removeEventListener('mouseup', upHandler);
      window.removeEventListener('touchmove', moveHandler);
      window.removeEventListener('touchend', upHandler);
    };
  }, [isDragging, handleMouseMove, handleMouseUp]);

  const pathData = useMemo(() => {
    if (!pathPoints || pathPoints.length < 2) return "";
    return `M ${pathPoints[0][0].toFixed(3)} ${pathPoints[0][1].toFixed(3)} ` +
           pathPoints.slice(1).map(p => `L ${p[0].toFixed(3)} ${p[1].toFixed(3)}`).join(' ') +
           " Z";
  }, [pathPoints]);

  const magnitude = Math.sqrt(x*x + y*y);
  const angleRad = Math.atan2(y, x);
  const angleDeg = (angleRad * 180 / Math.PI);

  return (
    <div style={{ textAlign: 'center', margin: '5px', padding: '10px', border: '1px solid #ccc', borderRadius: '8px', background: 'white', width: '220px' }}>
      <h5 style={{marginTop: 0, marginBottom: '5px'}}>{title}</h5>
      <svg
        ref={svgRef}
        viewBox={`${viewBox.minX} ${viewBox.minY} ${viewBox.width} ${viewBox.height}`}
        width="180" height="180"
        style={{ cursor: isDragging ? 'grabbing' : 'default', backgroundColor: bgColor, display: 'block', margin: '0 auto', touchAction: 'none' }}
        onMouseUp={handleMouseUp} onTouchEnd={handleMouseUp}
      >
        <line x1={viewBox.minX} y1="0" x2={viewBox.minX + viewBox.width} y2="0" stroke={axisColor} strokeWidth="0.02" />
        <line x1="0" y1={viewBox.minY} x2="0" y2={viewBox.minY + viewBox.height} stroke={axisColor} strokeWidth="0.02" />
        <text x={viewBox.minX + viewBox.width - 0.1} y="-0.1" fontSize="0.15" fill={axisColor} textAnchor="end">{xLabel}</text>
        <text x="-0.1" y={viewBox.minY + 0.15} fontSize="0.15" fill={axisColor} textAnchor="end" dominantBaseline="hanging">{yLabel}</text>
        {[ -1, 1].map(tick => ( <g key={`tick-${tick}`}> {Math.abs(tick) < 1.5 && <> <line x1={tick} y1="-0.05" x2={tick} y2="0.05" stroke={axisColor} strokeWidth="0.02" /> <line x1="-0.05" y1={tick} x2="0.05" y2={tick} stroke={axisColor} strokeWidth="0.02" /> </>} </g> ))}
        {maxRadius === 1 && <circle cx="0" cy="0" r="1" fill="none" stroke={axisColor} strokeWidth="0.015" strokeDasharray="0.05,0.05" />}
        <path d={pathData} fill="none" stroke={pathColor} strokeWidth="0.025" strokeDasharray="0.05, 0.05" pointerEvents="none" />
        <line x1="0" y1="0" x2={x} y2={y} stroke={pointColor} strokeWidth="0.03" opacity="0.6" />
        <circle id={`${pointId}-handle`} cx={x} cy={y} r="0.25" fill="transparent" style={{ cursor: 'grab' }} onMouseDown={handleMouseDown} onTouchStart={handleMouseDown} />
        <circle id={pointId} cx={x} cy={y} r="0.1" fill={pointColor} stroke="black" strokeWidth="0.01" pointerEvents="none" />
        <circle cx={x} cy={y} r="0.03" fill="black" pointerEvents="none" />
      </svg>
      <p style={{fontFamily: 'monospace', fontSize: '0.8em', marginTop: '5px', marginBottom: '0', lineHeight: '1.3'}}>
          Mag: {magnitude.toFixed(3)} <br/>
          Phase: {angleDeg.toFixed(1)}°
      </p>
    </div>
  );
};

/**
 * BlochSphere3D Component (Unchanged)
 */
const BlochSphere3D = ({ blochX, blochY, blochZ }) => {
    const mountRef = useRef(null);
    const stateVectorRef = useRef();
    const sceneRef = useRef();
    const rendererRef = useRef();
    const labelRendererRef = useRef();
    const controlsRef = useRef();
    const animationFrameIdRef = useRef();
    const timeoutIdRef = useRef();

    useEffect(() => {
        clearTimeout(timeoutIdRef.current);
        timeoutIdRef.current = setTimeout(() => {
            let errorMessage = "";
            if (typeof window.THREE === 'undefined') errorMessage = "Error: THREE.js core library not found.";
            else if (typeof window.THREE.OrbitControls === 'undefined') errorMessage = "Error: THREE.OrbitControls not found.";
            else if (typeof window.THREE.CSS2DRenderer === 'undefined') errorMessage = "Error: THREE.CSS2DRenderer not found.";

            if (errorMessage) {
                console.error(errorMessage);
                if (mountRef.current) mountRef.current.innerHTML = `<p style='color: red; padding: 10px;'>${errorMessage}</p>`;
                return;
            }

            const THREE = window.THREE;
            const scene = new THREE.Scene();
            sceneRef.current = scene;
            scene.background = new THREE.Color(0xf8f8f8);
            const width = 250; const height = 250;
            const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
            camera.position.set(1.8, 1.8, 2.5); camera.lookAt(scene.position);
            const renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(width, height); rendererRef.current = renderer;
             if (mountRef.current) { mountRef.current.innerHTML = ''; mountRef.current.appendChild(renderer.domElement); }
             else { return; }
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6); scene.add(ambientLight);
            const pointLight = new THREE.PointLight(0xffffff, 0.8); pointLight.position.set(5, 5, 5); scene.add(pointLight);
            const sphereGeometry = new THREE.SphereGeometry(1, 32, 32);
            const sphereMaterial = new THREE.MeshPhongMaterial({ color: 0xcccccc, transparent: true, opacity: 0.2, shininess: 50 });
            const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial); scene.add(sphere);
            const wireframeGeometry = new THREE.WireframeGeometry(sphereGeometry);
            const wireframeMaterial = new THREE.LineBasicMaterial({ color: 0xaaaaaa, linewidth: 1, opacity: 0.3, transparent: true });
            const wireframe = new THREE.LineSegments(wireframeGeometry, wireframeMaterial); scene.add(wireframe);
            const createAxis = (direction, color) => { /* ... */
                const material = new THREE.LineBasicMaterial({ color: color, linewidth: 2 });
                const points = [new THREE.Vector3(0, 0, 0), direction.multiplyScalar(1.2)];
                const geometry = new THREE.BufferGeometry().setFromPoints(points);
                return new THREE.Line(geometry, material);
            };
            scene.add(createAxis(new THREE.Vector3(1, 0, 0), 0xff0000)); scene.add(createAxis(new THREE.Vector3(0, 1, 0), 0x00ff00)); scene.add(createAxis(new THREE.Vector3(0, 0, 1), 0x0000ff));
            const labelRenderer = new THREE.CSS2DRenderer(); labelRendererRef.current = labelRenderer;
            labelRenderer.setSize(width, height); labelRenderer.domElement.style.position = 'absolute';
            labelRenderer.domElement.style.top = '0px'; labelRenderer.domElement.style.pointerEvents = 'none';
             if (mountRef.current) { mountRef.current.appendChild(labelRenderer.domElement); }
             else { return; }
            const createLabel = (text, position) => { /* ... */
                 const div = document.createElement('div'); div.style.position = 'absolute'; div.style.color = 'black';
                 div.style.fontSize = '11px'; div.style.fontFamily = 'sans-serif'; div.textContent = text; div.style.pointerEvents = 'none';
                 const label = new THREE.CSS2DObject(div); label.position.copy(position); scene.add(label); return label;
            };
            createLabel('+X', new THREE.Vector3(1.3, 0, 0)); createLabel('+Y', new THREE.Vector3(0, 1.3, 0));
            createLabel('|0⟩ (+Z)', new THREE.Vector3(0, 0, 1.3)); createLabel('|1⟩ (-Z)', new THREE.Vector3(0, 0, -1.3));
            const arrowOrigin = new THREE.Vector3(0, 0, 0);
            const initialDirection = new THREE.Vector3(blochX, blochY, blochZ).lengthSq() > 1e-9 ? new THREE.Vector3(blochX, blochY, blochZ).normalize() : new THREE.Vector3(0, 0, 1);
            const arrowLength = 1.0; const arrowColor = 0xff8c00;
            const arrowHelper = new THREE.ArrowHelper(initialDirection, arrowOrigin, arrowLength, arrowColor, 0.15, 0.08);
            scene.add(arrowHelper); stateVectorRef.current = arrowHelper;
            const controls = new THREE.OrbitControls(camera, renderer.domElement); controlsRef.current = controls;
            controls.enablePan = false; controls.minDistance = 2; controls.maxDistance = 5;
            controls.enableDamping = true; controls.dampingFactor = 0.1;
            const animate = () => { animationFrameIdRef.current = requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); labelRenderer.render(scene, camera); };
            animate();
        }, 100); // Delay

        return () => { // Cleanup
            clearTimeout(timeoutIdRef.current); cancelAnimationFrame(animationFrameIdRef.current);
            if (controlsRef.current) controlsRef.current.dispose();
            if (rendererRef.current) rendererRef.current.dispose();
            if (mountRef.current) mountRef.current.innerHTML = '';
        };
    }, []); // Mount/unmount effect

    useEffect(() => { // Update arrow effect
        if (typeof window.THREE !== 'undefined' && stateVectorRef.current) {
            const THREE = window.THREE;
            const newDirection = new THREE.Vector3(blochX, blochY, blochZ);
            if (newDirection.lengthSq() > 1e-9) newDirection.normalize(); else newDirection.set(0, 0, 1);
            stateVectorRef.current.setDirection(newDirection);
        }
    }, [blochX, blochY, blochZ]);

    const containerStyle = { width: '250px', height: '250px', position: 'relative', margin: '0 auto', border: '1px solid #ccc', borderRadius: '8px', overflow: 'hidden' };
    return <div ref={mountRef} style={containerStyle} />;
};

/**
 * EquivalenceClassFan Component (Unchanged)
 */
const EquivalenceClassFan = ({ alphaPairs, betaPairs, numSteps }) => {
    const viewBox = { minX: -1.5, minY: -1.5, width: 3, height: 3 };
    const axisColor = '#999';
    const getAlphaColorForStep = (step) => { const hue = (step / numSteps) * 90; return `hsl(${hue}, 90%, 55%)`; };
    const getBetaColorForStep = (step) => { const hue = 180 + (step / numSteps) * 120; return `hsl(${hue}, 80%, 60%)`; };

    return (
        <div style={{ textAlign: 'center', margin: '5px', padding: '10px', border: '1px solid #ccc', borderRadius: '8px', background: 'white', width: '220px' }}>
            <h5 style={{marginTop: 0, marginBottom: '5px'}}>Equivalence Class Fan</h5>
            <svg viewBox={`${viewBox.minX} ${viewBox.minY} ${viewBox.width} ${viewBox.height}`} width="180" height="180" style={{ backgroundColor: '#f8f8f8', display: 'block', margin: '0 auto' }} >
                <line x1={viewBox.minX} y1="0" x2={viewBox.minX + viewBox.width} y2="0" stroke={axisColor} strokeWidth="0.02" />
                <line x1="0" y1={viewBox.minY} x2="0" y2={viewBox.minY + viewBox.height} stroke={axisColor} strokeWidth="0.02" />
                {alphaPairs.length > 0 && <circle cx="0" cy="0" r={Math.sqrt(alphaPairs[0][0]**2 + alphaPairs[0][1]**2)} fill="none" stroke="rgba(255, 127, 80, 0.3)" strokeWidth="0.015" strokeDasharray="0.05,0.05" />}
                {betaPairs.length > 0 && <circle cx="0" cy="0" r={Math.sqrt(betaPairs[0][0]**2 + betaPairs[0][1]**2)} fill="none" stroke="rgba(135, 206, 235, 0.3)" strokeWidth="0.015" strokeDasharray="0.05,0.05" />}
                {alphaPairs.map((alphaPair, index) => {
                    const betaPair = betaPairs[index]; const alphaColor = getAlphaColorForStep(index); const betaColor = getBetaColorForStep(index);
                    if (!alphaPair || !betaPair) return null;
                    return ( <g key={`pair-${index}`}> <line x1="0" y1="0" x2={alphaPair[0]} y2={alphaPair[1]} stroke={alphaColor} strokeWidth="0.025" opacity="0.7" /> <line x1="0" y1="0" x2={betaPair[0]} y2={betaPair[1]} stroke={betaColor} strokeWidth="0.025" opacity="0.7" /> </g> );
                })}
            </svg>
             <p style={{fontFamily: 'monospace', fontSize: '0.8em', marginTop: '5px', marginBottom: '0', lineHeight: '1.3', color: '#555'}}> α' fan (warm colors) <br/> β' fan (cool colors) </p>
        </div>
    );
};


/**
 * QubitBlochOriginViz Component (Main - Corrected Drag Logic)
 */
const QubitBlochOriginViz = () => {
  // State for the *base* parameters
  const [aBase, setABase] = useState(1.0);
  const [bBase, setBBase] = useState(0.0);
  const [cBase, setCBase] = useState(0.0);
  const [dBase, setDBase] = useState(0.0);
  // State for the global phase angle gamma (slider value)
  const [gamma, setGamma] = useState(0.0);

  const epsilon = 1e-9;
  const numPathSteps = 48;

  // --- Rotation Function ---
  const rotateComplex = useCallback((x, y, angle) => {
    const cosG = Math.cos(angle); const sinG = Math.sin(angle);
    const xRot = x * cosG - y * sinG; const yRot = x * sinG + y * cosG;
    return [xRot, yRot];
  }, []);

  // --- *** CORRECTED Drag Handlers *** ---

  /**
   * Handles dragging the alpha plot point.
   * The dragged coordinates (newARot, newBRot) are interpreted as the target
   * for the base alpha directly. Normalization adjusts base beta.
   */
  const handleDragAlpha = useCallback((newA, newB) => { // Input coords are now treated as target base coords
    let targetABase = newA;
    let targetBBase = newB;
    let radiusAlphaBaseSq = targetABase * targetABase + targetBBase * targetBBase;

    // Clamp magnitude to max 1
    if (radiusAlphaBaseSq > 1.0) {
        const r = Math.sqrt(radiusAlphaBaseSq);
        targetABase /= r;
        targetBBase /= r;
        radiusAlphaBaseSq = 1.0;
    }

    // Calculate required magnitude for base beta
    const radiusBetaBaseSq = 1.0 - radiusAlphaBaseSq;
    const nextMagBetaBase = Math.sqrt(Math.max(0, radiusBetaBaseSq));

    // Preserve the *current* phase of base beta
    const currentPhaseBetaBase = Math.atan2(dBase, cBase); // Use existing dBase, cBase

    // Calculate new base beta
    const nextCBase = nextMagBetaBase * Math.cos(currentPhaseBetaBase);
    const nextDBase = nextMagBetaBase * Math.sin(currentPhaseBetaBase);

    // Update state with the new base values
    setABase(targetABase);
    setBBase(targetBBase);
    setCBase(nextCBase);
    setDBase(nextDBase);

  }, [cBase, dBase]); // Depends on current base beta phase

  /**
   * Handles dragging the beta plot point.
   * Symmetric logic: dragged coords set target base beta, normalization
   * adjusts base alpha while preserving its phase.
   */
  const handleDragBeta = useCallback((newC, newD) => { // Input coords are now treated as target base coords
    let targetCBase = newC;
    let targetDBase = newD;
    let radiusBetaBaseSq = targetCBase * targetCBase + targetDBase * targetDBase;

    // Clamp magnitude to max 1
    if (radiusBetaBaseSq > 1.0) {
        const r = Math.sqrt(radiusBetaBaseSq);
        targetCBase /= r;
        targetDBase /= r;
        radiusBetaBaseSq = 1.0;
    }

    // Calculate required magnitude for base alpha
    const radiusAlphaBaseSq = 1.0 - radiusBetaBaseSq;
    const nextMagAlphaBase = Math.sqrt(Math.max(0, radiusAlphaBaseSq));

    // Preserve the *current* phase of base alpha
    const currentPhaseAlphaBase = Math.atan2(bBase, aBase); // Use existing bBase, aBase

    // Calculate new base alpha
    const nextABase = nextMagAlphaBase * Math.cos(currentPhaseAlphaBase);
    const nextBBase = nextMagAlphaBase * Math.sin(currentPhaseAlphaBase);

    // Update state with the new base values
    setABase(nextABase);
    setBBase(nextBBase);
    setCBase(targetCBase);
    setDBase(targetDBase);

  }, [aBase, bBase]); // Depends on current base alpha phase


  // --- Calculations (Unchanged) ---
  // Calculate current rotated coordinates for display based on gamma slider
  const [aRot, bRot] = rotateComplex(aBase, bBase, gamma);
  const [cRot, dRot] = rotateComplex(cBase, dBase, gamma);

  // Calculate path points for dashed circles and fan plot
  const { alphaPath, betaPath } = useMemo(() => {
    const alphaPts = []; const betaPts = [];
    for (let i = 0; i <= numPathSteps; i++) {
        const pathGamma = i * (2 * Math.PI / numPathSteps);
        alphaPts.push(rotateComplex(aBase, bBase, pathGamma));
        betaPts.push(rotateComplex(cBase, dBase, pathGamma));
    }
    return { alphaPath: alphaPts, betaPath: betaPts };
  }, [aBase, bBase, cBase, dBase, rotateComplex, numPathSteps]);

  // Calculate invariant parameters and Bloch coordinates
  const magAlphaBase = Math.sqrt(aBase*aBase + bBase*bBase);
  const theta = 2 * Math.acos(Math.min(magAlphaBase, 1.0));
  const phaseAlphaBase = Math.atan2(bBase, aBase);
  const phaseBetaBase = Math.atan2(dBase, cBase);
  let relativePhi = phaseBetaBase - phaseAlphaBase;
  while (relativePhi <= -Math.PI) relativePhi += 2 * Math.PI;
  while (relativePhi > Math.PI) relativePhi -= 2 * Math.PI;
  const blochX = Math.sin(theta) * Math.cos(relativePhi);
  const blochY = Math.sin(theta) * Math.sin(relativePhi);
  const blochZ = Math.cos(theta);
  const sumOfSquaresBase = aBase*aBase + bBase*bBase + cBase*cBase + dBase*dBase;

  return (
    // --- JSX Structure (Unchanged) ---
    <div style={{ fontFamily: 'Inter, sans-serif', padding: '15px', maxWidth: '1000px', margin: 'auto' }}>
      <h2 style={{ textAlign: 'center', marginBottom: '15px', color: '#333' }}>Visualizing the Bloch Sphere Origin</h2>
      <p style={{ lineHeight: '1.5', color: '#555', fontSize: '0.9em' }}>
        Drag points for α' and β' (solid dots) to set the underlying physical state (base state). Normalization links α and β. The dashed circles show the **Equivalence Class** path traced by applying a global phase γ.
        The **Equivalence Class Fan** plot shows all pairs (α', β') for this class simultaneously. The Bloch Sphere point represents this entire class.
      </p>
       <div style={{ margin: '15px 0', padding: '10px 15px', border: '1px solid #e0e0e0', borderRadius: '8px', background: '#f9f9f9' }}>
           <label htmlFor="gamma-slider" style={{ display: 'block', marginBottom: '5px', fontWeight: '500', color: '#333', fontSize: '0.9em' }}>
               Global Phase γ: {(gamma * 180 / Math.PI).toFixed(1)}° (Moves solid dot along dashed path)
            </label>
           <input type="range" id="gamma-slider" min="0" max={2 * Math.PI} step="0.01" value={gamma}
                onChange={(e) => setGamma(parseFloat(e.target.value))} style={{ width: '100%' }} />
       </div>
       <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', alignItems: 'flex-start', gap: '10px', marginBottom: '20px' }}>
            {/* Pass rotated coords for display, but drag handler sets base state */}
            <ComplexPlanePlot
              x={aRot} y={bRot} onDrag={handleDragAlpha} maxRadius={1}
              xLabel="Re α'" yLabel="Im α'" title="α' = e^(iγ) α"
              pointId="alpha-point" pointColor="coral" bgColor="#fff5f0"
              pathPoints={alphaPath}
            />
            <ComplexPlanePlot
              x={cRot} y={dRot} onDrag={handleDragBeta} maxRadius={1}
              xLabel="Re β'" yLabel="Im β'" title="β' = e^(iγ) β"
              pointId="beta-point" pointColor="skyblue" bgColor="#f0f9ff"
              pathPoints={betaPath}
            />
            <EquivalenceClassFan
                alphaPairs={alphaPath}
                betaPairs={betaPath}
                numSteps={numPathSteps}
            />
            <div style={{ textAlign: 'center', margin: '5px', padding: '10px', border: '1px solid #ccc', borderRadius: '8px', background: 'white', width: '274px' }}>
                 <h5 style={{marginTop: 0, marginBottom: '5px'}}>Bloch Sphere</h5>
                 <BlochSphere3D blochX={blochX} blochY={blochY} blochZ={blochZ} />
            </div>
       </div>
       <div style={{ padding: '15px', border: '1px solid #e0e0e0', borderRadius: '8px', background: '#f9f9f9', fontSize: '0.9em' }}>
            <h4 style={{ marginTop: '0', marginBottom: '10px', color: '#333' }}>Invariant Parameters & Bloch Coordinates:</h4>
            <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-around', gap: '10px' }}>
                <p style={{fontFamily: 'monospace', lineHeight: '1.6', color: '#333', margin: 0 }}>
                   θ = 2arccos(|α<sub>base</sub>|) = {(theta * 180 / Math.PI).toFixed(1)}° <br/>
                   φ = arg(β<sub>base</sub>)-arg(α<sub>base</sub>) = {(relativePhi * 180 / Math.PI).toFixed(1)}°
                </p>
                <p style={{fontFamily: 'monospace', lineHeight: '1.6', color: '#28a745', margin: 0, fontWeight: '500' }}>
                   Bloch X = sinθ cosφ = {blochX.toFixed(3)} <br/>
                   Bloch Y = sinθ sinφ = {blochY.toFixed(3)} <br/>
                   Bloch Z = cosθ = {blochZ.toFixed(3)}
                </p>
           </div>
            <p style={{fontFamily: 'monospace', fontSize: '0.9em', marginTop: '10px', color: '#666' }}>
               Base State: α = {aBase.toFixed(3)} {bBase >= 0 ? '+' : '-'} {Math.abs(bBase).toFixed(3)}i,&nbsp;
               β = {cBase.toFixed(3)} {dBase >= 0 ? '+' : '-'} {Math.abs(dBase).toFixed(3)}i
               &nbsp;(Norm: {sumOfSquaresBase.toFixed(3)})
           </p>
       </div>
    </div>
  );
};

export default QubitBlochOriginViz;
