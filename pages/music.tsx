import { useRef, useState, useEffect, useCallback } from 'react';
import { Nav } from '@/components/nav';
import { useMusic } from '@/components/music-provider';

export default function MusicPage() {
  const music = useMusic();
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
  const animationRef = useRef<number>(0);
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 });
  const [dropError, setDropError] = useState<string | null>(null);

  // ---------- Canvas sizing (client-only, with DPR) ----------
  useEffect(() => {
    const update = () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      setCanvasSize({ width, height });
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.floor(width * ratio);
      canvas.height = Math.floor(height * ratio);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      const ctx = canvas.getContext('2d');
      if (ctx) ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    };
    update();
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, []);

  // ---------- Audio setup ----------
  useEffect(() => {
    if (!file) return;

    // Clean up previous audio instance
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = '';
      audioRef.current = null;
    }

    const audio = new Audio();
    audioRef.current = audio;

    const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
    const source = ctx.createMediaElementSource(audio);
    const analyserNode = ctx.createAnalyser();
    analyserNode.fftSize = 256;
    source.connect(analyserNode);
    analyserNode.connect(ctx.destination);
    setAnalyser(analyserNode);

    audio.src = URL.createObjectURL(file);
    audio.play().catch(console.error);

    return () => {
      audio.pause();
      ctx.close();
      URL.revokeObjectURL(audio.src);
    };
  }, [file]);

  // ---------- Animation loop ----------
  useEffect(() => {
    if (!analyser) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const draw = () => {
      animationRef.current = requestAnimationFrame(draw);
      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      analyser.getByteFrequencyData(dataArray);

      ctx.clearRect(0, 0, canvasSize.width, canvasSize.height);

      const barWidth = (canvasSize.width / bufferLength) * 2.5;
      let x = 0;
      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * canvasSize.height;
        ctx.fillStyle = `rgb(${Math.floor((dataArray[i] / 255) * 200)}, 50, 200)`;
        ctx.fillRect(x, canvasSize.height - barHeight, barWidth, barHeight);
        x += barWidth + 1;
      }
    };

    draw();
    return () => cancelAnimationFrame(animationRef.current);
  }, [analyser, canvasSize.width, canvasSize.height]);

  // ---------- Drag-and-drop handlers ----------
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDropError(null);
    const items = e.dataTransfer.files;
    if (items.length && items[0].type.startsWith('audio/')) {
      setFile(items[0]);
    } else {
      setDropError('Please drop a valid audio file.');
      window.setTimeout(() => setDropError(null), 3000);
    }
  };

  const clearFile = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = '';
      audioRef.current = null;
    }
    setFile(null);
    setAnalyser(null);
    cancelAnimationFrame(animationRef.current);
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, canvasSize.width, canvasSize.height);
    }
  }, [canvasSize.width, canvasSize.height]);

  return (
    <div className="page music-page" onDragOver={handleDragOver} onDrop={handleDrop}>
      <Nav />

      {/* Drop zone overlay */}
      {!file && (
        <div className="drop-zone">
          <p>Drag & drop an MP3 file here to play</p>
          <p className="hint">(or use the controls below for YouTube/local tracks)</p>
          <p className="hint">Audio stays local to your device — nothing is uploaded.</p>
        </div>
      )}

      {dropError && (
        <div className="drop-zone" style={{ background: 'rgba(80,0,0,0.6)' }}>
          <p>{dropError}</p>
        </div>
      )}

      <p className="page-kicker">Global audio</p>
      <h1 className="page-title">Music</h1>

      {/* Interactive background canvas */}
      <canvas
        ref={canvasRef}
        id="bg-canvas"
        className="bg-canvas"
        width={canvasSize.width}
        height={canvasSize.height}
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: canvasSize.width,
          height: canvasSize.height,
          zIndex: 1,
          pointerEvents: 'none',
        }}
      />

      <section className="panel">
        <p>
          The music engine is global. Start a local track or a YouTube playlist
          here, then move through the app while the orb keeps reacting.
        </p>

        {file && (
          <div style={{ marginBottom: '0.75rem' }}>
            <span className="badge badge-running" style={{ marginRight: '0.5rem' }}>
              Local: {file.name}
            </span>
            <button className="button" onClick={clearFile} type="button">
              Clear file
            </button>
          </div>
        )}

        <div className="music-controls">
          <button className="button" onClick={() => void music.previous()} type="button">
            Previous
          </button>
          <button className="button" onClick={() => void music.toggle()} type="button">
            {music.isPlaying ? 'Pause' : 'Play'}
          </button>
          <button className="button" onClick={() => void music.next()} type="button">
            Next
          </button>
          <select
            className="button"
            onChange={(event) => void music.select(event.target.value)}
            value={music.selectedTrack.id}
          >
            {music.tracks.map((track) => (
              <option key={track.id} value={track.id}>
                {track.label}
              </option>
            ))}
          </select>
        </div>

        <form
          className="youtube-form"
          onSubmit={(event) => {
            event.preventDefault();
            void music.loadYoutube();
          }}
        >
          <input
            aria-label="YouTube playlist or video URL"
            onChange={(event) => music.setYoutubeUrl(event.target.value)}
            placeholder="https://www.youtube.com/watch?v=... or playlist"
            value={music.youtubeUrl}
          />
          <button
            className="button"
            disabled={music.youtubeLoading || !music.youtubeUrl.trim()}
            type="submit"
          >
            {music.youtubeLoading ? 'Loading' : 'Load YouTube'}
          </button>
        </form>

        {music.youtubeEntries.length > 0 && (
          <div className="youtube-list">
            {music.youtubeEntries.slice(0, 8).map((entry) => (
              <span key={entry.id}>{entry.title}</span>
            ))}
          </div>
        )}

        <p>{music.status}</p>
      </section>
    </div>
  );
}
