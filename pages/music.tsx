import { Nav } from '@/components/nav';
import { useMusic } from '@/components/music-provider';

export default function MusicPage() {
  const music = useMusic();
  return (
    <div className="page">
      <Nav />
      <p className="page-kicker">Global audio</p>
      <h1 className="page-title">Music</h1>
      <section className="panel">
        <p>The music engine is global. Start a local track or a YouTube playlist here, then move through the app while the orb keeps reacting.</p>
        <div className="music-controls">
          <button className="button" onClick={() => void music.previous()} type="button">Previous</button>
          <button className="button" onClick={() => void music.toggle()} type="button">{music.isPlaying ? 'Pause' : 'Play'}</button>
          <button className="button" onClick={() => void music.next()} type="button">Next</button>
          <select className="button" onChange={(event) => void music.select(event.target.value)} value={music.selectedTrack.id}>
            {music.tracks.map((track) => <option key={track.id} value={track.id}>{track.label}</option>)}
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
          <button className="button" disabled={music.youtubeLoading || !music.youtubeUrl.trim()} type="submit">
            {music.youtubeLoading ? 'Loading' : 'Load YouTube'}
          </button>
        </form>
        {music.youtubeEntries.length > 0 && (
          <div className="youtube-list">
            {music.youtubeEntries.slice(0, 8).map((entry) => <span key={entry.id}>{entry.title}</span>)}
          </div>
        )}
        <p>{music.status}</p>
      </section>
    </div>
  );
}
