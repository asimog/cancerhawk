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
        <p>The music engine is global. Start it here, then move through Current Block, Previous Blocks, or Run Research while the orb keeps reacting.</p>
        <div className="music-controls">
          <button className="button" onClick={() => void music.previous()} type="button">Previous</button>
          <button className="button" onClick={() => void music.toggle()} type="button">{music.isPlaying ? 'Pause' : 'Play'}</button>
          <button className="button" onClick={() => void music.next()} type="button">Next</button>
          <select className="button" onChange={(event) => void music.select(event.target.value)} value={music.selectedTrack.id}>
            {music.tracks.map((track) => <option key={track.id} value={track.id}>{track.label}</option>)}
          </select>
        </div>
        <p>{music.status}</p>
      </section>
    </div>
  );
}
