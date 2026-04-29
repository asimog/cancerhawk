'use client';

import { useEffect, useRef } from 'react';
import { useMusic } from '@/components/music-provider';

export function SiteBackground() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const music = useMusic();
  const featuresRef = useRef(music.features);
  const pointerRef = useRef({ x: 0.5, y: 0.42, down: false, bloom: 0 });

  useEffect(() => {
    featuresRef.current = music.features;
  }, [music.features]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const canvasElement = canvas;
    const drawingContext = ctx;
    let raf = 0;
    let width = 0;
    let height = 0;
    let time = 0;

    function resize() {
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      width = window.innerWidth;
      height = window.innerHeight;
      canvasElement.width = Math.floor(width * ratio);
      canvasElement.height = Math.floor(height * ratio);
      canvasElement.style.width = `${width}px`;
      canvasElement.style.height = `${height}px`;
      drawingContext.setTransform(ratio, 0, 0, ratio, 0, 0);
    }

    function drawOrb(cx: number, cy: number, radius: number) {
      const audio = featuresRef.current;
      const ocean = drawingContext.createRadialGradient(cx - radius * 0.3, cy - radius * 0.35, radius * 0.08, cx, cy, radius);
      ocean.addColorStop(0, '#d9fffa');
      ocean.addColorStop(0.28, '#55c875');
      ocean.addColorStop(0.55, '#1e7fa0');
      ocean.addColorStop(1, '#061015');
      drawingContext.fillStyle = ocean;
      drawingContext.beginPath();
      drawingContext.arc(cx, cy, radius, 0, Math.PI * 2);
      drawingContext.fill();

      drawingContext.save();
      drawingContext.globalCompositeOperation = 'screen';
      drawingContext.shadowColor = `rgba(124,228,210,${0.3 + audio.volume * 0.5})`;
      drawingContext.shadowBlur = 20 + audio.bass * 60;
      drawingContext.strokeStyle = `rgba(124,228,210,${0.22 + audio.high * 0.35})`;
      drawingContext.lineWidth = 1.5;
      for (let i = 0; i < 5; i += 1) {
        drawingContext.beginPath();
        drawingContext.ellipse(cx, cy, radius * (1.1 + i * 0.12), radius * (0.36 + i * 0.035), time * 0.15 + i * 0.5, 0, Math.PI * 2);
        drawingContext.stroke();
      }
      drawingContext.restore();
    }

    function frame() {
      time += 0.016;
      const audio = featuresRef.current;
      const pointer = pointerRef.current;
      const energy = audio.isPlaying ? audio.volume : 0.08;
      const bass = audio.isPlaying ? audio.bass : 0.04;
      pointer.bloom *= 0.93;
      drawingContext.clearRect(0, 0, width, height);
      const bg = drawingContext.createLinearGradient(0, 0, 0, height);
      bg.addColorStop(0, '#020407');
      bg.addColorStop(0.5, '#03080b');
      bg.addColorStop(1, '#000000');
      drawingContext.fillStyle = bg;
      drawingContext.fillRect(0, 0, width, height);

      const idleX = 0.5 + Math.sin(time * 0.17) * 0.035;
      const idleY = 0.42 + Math.cos(time * 0.13) * 0.025;
      const targetX = pointer.down ? pointer.x : idleX * 0.72 + pointer.x * 0.28;
      const targetY = pointer.down ? pointer.y : idleY * 0.72 + pointer.y * 0.28;
      const cx = width * targetX;
      const cy = height * targetY;
      const radius = Math.min(width, height) * (0.11 + bass * 0.1 + pointer.bloom * 0.035);
      const bloom = drawingContext.createRadialGradient(cx, cy, radius * 0.2, cx, cy, Math.max(width, height) * 0.42);
      bloom.addColorStop(0, `rgba(73,197,182,${0.18 + energy * 0.22 + pointer.bloom * 0.16})`);
      bloom.addColorStop(0.42, `rgba(39,121,167,${0.08 + audio.mid * 0.18 + pointer.bloom * 0.09})`);
      bloom.addColorStop(1, 'rgba(0,0,0,0)');
      drawingContext.fillStyle = bloom;
      drawingContext.fillRect(0, 0, width, height);
      drawOrb(cx, cy, radius);

      drawingContext.save();
      drawingContext.globalCompositeOperation = 'screen';
      for (let i = 0; i < 210; i += 1) {
        const angle = (i / 170) * Math.PI * 2 + time * (0.08 + audio.high * 0.1);
        const lane = 1.5 + (i % 19) * 0.055 + energy * 0.28 + pointer.bloom * 0.18;
        const x = cx + Math.cos(angle) * radius * lane + Math.sin(time + i) * 8;
        const y = cy + Math.sin(angle * 1.3) * radius * lane * 0.55;
        drawingContext.fillStyle = i % 4 === 0 ? 'rgba(245,251,251,0.86)' : 'rgba(124,228,210,0.58)';
        drawingContext.beginPath();
        drawingContext.arc(x, y, 0.9 + (i % 5) * 0.25 + audio.high * 2, 0, Math.PI * 2);
        drawingContext.fill();
      }
      drawingContext.restore();

      if (audio.beat) {
        drawingContext.strokeStyle = 'rgba(245,251,251,0.72)';
        drawingContext.lineWidth = 2;
        drawingContext.beginPath();
        drawingContext.arc(cx, cy, radius * 1.45, 0, Math.PI * 2);
        drawingContext.stroke();
      }

      raf = requestAnimationFrame(frame);
    }

    resize();
    function updatePointer(event: PointerEvent) {
      pointerRef.current.x = event.clientX / Math.max(1, window.innerWidth);
      pointerRef.current.y = event.clientY / Math.max(1, window.innerHeight);
    }
    function pointerDown(event: PointerEvent) {
      updatePointer(event);
      pointerRef.current.down = true;
      pointerRef.current.bloom = 1;
    }
    function pointerUp() {
      pointerRef.current.down = false;
    }
    function pointerMove(event: PointerEvent) {
      updatePointer(event);
      if (pointerRef.current.down) pointerRef.current.bloom = Math.min(1, pointerRef.current.bloom + 0.08);
    }
    window.addEventListener('resize', resize);
    window.addEventListener('pointerdown', pointerDown);
    window.addEventListener('pointerup', pointerUp);
    window.addEventListener('pointermove', pointerMove);
    raf = requestAnimationFrame(frame);
    return () => {
      window.removeEventListener('resize', resize);
      window.removeEventListener('pointerdown', pointerDown);
      window.removeEventListener('pointerup', pointerUp);
      window.removeEventListener('pointermove', pointerMove);
      cancelAnimationFrame(raf);
    };
  }, []);

  return (
    <div className="site-background-shell" aria-hidden="true">
      <canvas className="site-background-canvas" ref={canvasRef} />
      <div className="site-background-grid" />
      <div className="site-background-vignette" />
    </div>
  );
}
