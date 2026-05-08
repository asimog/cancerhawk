import { useCallback, useRef, useState } from 'react';
import { Nav } from '@/components/nav';
import { useMusic } from '@/components/music-provider';

export default function MusicPage() {
  const music = useMusic();
  const [file, setFile] = useState<File | null>(null);
  const [dropError, setDropError] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const dropRef = useRef<HTMLDivElement | null>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    if (dropRef.current && !dropRef.current.contains(e.relatedTarget as Node)) {
      setIsDragOver(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    setDropError(null);
    const items = e.dataTransfer.files;
    if (items.length && items[0].type.startsWith('audio/')) {
      setFile(items[0]);
      void music.playFile(items[0]).catch((error) => {
        setDropError(error instanceof Error ? error.message : 'The audio file could not be played.');
      });
    } else {
      setDropError('Please drop a valid audio file (MP3, WAV, etc.).');
      window.setTimeout(() => setDropError(null), 3000);
    }
  };

  const clearFile = useCallback(() => {
    setFile(null);
  }, []);

  return (
    <div className="page music-page">
      <Nav />

      <p className="page-kicker">Global audio</p>
      <h1 className="page-title">Music</h1>

      <p className="muted" style={{ marginBottom: 24 }}>
        The music engine is global — start a local track or YouTube playlist here, then browse the app while the orb and particles stay audio-reactive.
      </p>

      {/* Two-option layout: Drag-and-drop + YouTube */}
      <div className="music-two-columns" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 24 }}>
        {/* Drag-and-drop column */}
        <div
          className={`panel music-drop-panel ${isDragOver ? 'music-drop-active' : ''}`}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          ref={dropRef}
          style={{
            border: `2px dashed ${isDragOver ? '#4caf50' : '#333'}`,
            borderRadius: 12,
            padding: 24,
            textAlign: 'center',
            transition: 'border-color 0.2s',
            minHeight: 140,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            gap: 8,
          }}
        >
          <p style={{ fontWeight: 600, margin: 0 }}>Drag & drop MP3</p>
          <p style={{ fontSize: 13, color: '#888', margin: 0 }}>Audio stays local — nothing uploaded.</p>
          {file && (
            <div style={{ marginTop: 8, display: 'flex', gap: 8, alignItems: 'center' }}>
              <span className="badge badge-running" style={{ fontSize: 12 }}>{file.name}</span>
              <button className="button" onClick={clearFile} type="button" style={{ fontSize: 12, padding: '2px 8px' }}>
                Clear
              </button>
            </div>
          )}
          {dropError && (
            <p style={{ color: '#ff6b6b', fontSize: 13, margin: 0 }}>{dropError}</p>
          )}
        </div>

        {/* YouTube column */}
        <div className="panel" style={{ borderRadius: 12, padding: 24, minHeight: 140 }}>
          <p style={{ fontWeight: 600, margin: 0, textAlign: 'center', marginBottom: 12 }}>Load YouTube</p>
          <form
            onSubmit={(event) => {
              event.preventDefault();
              void music.loadYoutube();
            }}
            style={{ display: 'flex', gap: 8 }}
          >
            <input
              aria-label="YouTube playlist or video URL"
              onChange={(event) => music.setYoutubeUrl(event.target.value)}
              placeholder="https://www.youtube.com/watch?v=..."
              value={music.youtubeUrl}
              style={{ flex: 1 }}
            />
            <button
              className="button"
              disabled={music.youtubeLoading || !music.youtubeUrl.trim()}
              type="submit"
            >
              {music.youtubeLoading ? 'Loading' : 'Load'}
            </button>
          </form>
          {music.youtubeEntries.length > 0 && (
            <div className="youtube-list" style={{ marginTop: 8, fontSize: 12, color: '#888' }}>
              {music.youtubeEntries.slice(0, 6).map((entry) => (
                <span key={entry.id}>{entry.title}</span>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Playback controls */}
      <section className="panel" style={{ borderRadius: 12, padding: 20 }}>
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
        <p style={{ fontSize: 13, color: '#888', margin: '8px 0 0 0' }}>{music.status}</p>
      </section>
    </div>
  );
}
