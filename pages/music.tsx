import { useCallback, useState } from 'react';
import { Nav } from '@/components/nav';
import { useMusic } from '@/components/music-provider';

export default function MusicPage() {
  const music = useMusic();
  const [file, setFile] = useState<File | null>(null);
  const [dropError, setDropError] = useState<string | null>(null);

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
      void music.playFile(items[0]).catch((error) => {
        setDropError(error instanceof Error ? error.message : 'The audio file could not be played.');
      });
    } else {
      setDropError('Please drop a valid audio file.');
      window.setTimeout(() => setDropError(null), 3000);
    }
  };

  const clearFile = useCallback(() => {
    setFile(null);
  }, []);

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
