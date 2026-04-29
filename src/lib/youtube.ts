export type YoutubeResolvedEntry = {
  id: string;
  title: string;
  videoId: string;
  watchUrl: string;
  embedUrl: string;
  thumbnailUrl: string | null;
  durationSeconds: number | null;
};

export type YoutubeResolvedMedia = {
  kind: 'video' | 'playlist';
  sourceUrl: string;
  title: string;
  uploader: string | null;
  thumbnailUrl: string | null;
  embedUrl: string;
  videoId: string | null;
  playlistId: string | null;
  entries: YoutubeResolvedEntry[];
  resolvedWith: 'youtube-dl' | 'oembed';
  warning: string | null;
};

type YoutubeDlEntry = {
  id?: string;
  title?: string;
  url?: string;
  webpage_url?: string;
  thumbnail?: string;
  duration?: number;
};

type YoutubeDlPayload = YoutubeDlEntry & {
  _type?: string;
  uploader?: string;
  channel?: string;
  entries?: YoutubeDlEntry[];
};

const runners = [
  process.env.YOUTUBE_DL_BINARY?.trim(),
  'youtube-dl',
  'yt-dlp',
].filter((value): value is string => Boolean(value));

function trim(value?: string | null) {
  const out = value?.trim();
  return out || null;
}

export function isYouTubeUrl(value: string) {
  try {
    const host = new URL(value).hostname.toLowerCase().replace(/^www\./, '');
    return host === 'youtube.com' || host.endsWith('.youtube.com') || host === 'youtu.be';
  } catch {
    return false;
  }
}

export function extractYouTubeVideoId(value: string) {
  try {
    const url = new URL(value);
    const host = url.hostname.toLowerCase().replace(/^www\./, '');
    const parts = url.pathname.split('/').filter(Boolean);
    if (host === 'youtu.be') return trim(parts[0]);
    const queryId = trim(url.searchParams.get('v'));
    if (queryId) return queryId;
    if (['embed', 'shorts', 'live'].includes(parts[0])) return trim(parts[1]);
    return null;
  } catch {
    return null;
  }
}

export function extractYouTubePlaylistId(value: string) {
  try {
    return trim(new URL(value).searchParams.get('list'));
  } catch {
    return null;
  }
}

function watchUrl(videoId: string, playlistId?: string | null) {
  const url = new URL(`https://www.youtube.com/watch?v=${encodeURIComponent(videoId)}`);
  if (playlistId) url.searchParams.set('list', playlistId);
  return url.toString();
}

function embedUrl(videoId?: string | null, playlistId?: string | null) {
  const url = new URL(videoId ? `https://www.youtube.com/embed/${encodeURIComponent(videoId)}` : 'https://www.youtube.com/embed/videoseries');
  if (playlistId) url.searchParams.set('list', playlistId);
  url.searchParams.set('autoplay', '1');
  url.searchParams.set('playsinline', '1');
  url.searchParams.set('rel', '0');
  url.searchParams.set('modestbranding', '1');
  return url.toString();
}

function thumbnail(videoId: string) {
  return `https://i.ytimg.com/vi/${encodeURIComponent(videoId)}/hqdefault.jpg`;
}

function normalizeEntry(entry: YoutubeDlEntry, playlistId: string | null, index: number): YoutubeResolvedEntry | null {
  const videoId = trim(entry.id) || (entry.url ? extractYouTubeVideoId(entry.url) : null) || (entry.webpage_url ? extractYouTubeVideoId(entry.webpage_url) : null);
  if (!videoId) return null;
  return {
    id: `${videoId}-${index}`,
    title: trim(entry.title) || `Video ${index + 1}`,
    videoId,
    watchUrl: trim(entry.webpage_url) || trim(entry.url) || watchUrl(videoId, playlistId),
    embedUrl: embedUrl(videoId, playlistId),
    thumbnailUrl: trim(entry.thumbnail) || thumbnail(videoId),
    durationSeconds: typeof entry.duration === 'number' ? entry.duration : null,
  };
}

function normalize(sourceUrl: string, payload: YoutubeDlPayload): YoutubeResolvedMedia {
  const sourceVideoId = extractYouTubeVideoId(sourceUrl);
  const sourcePlaylistId = extractYouTubePlaylistId(sourceUrl);
  const playlistId = payload._type === 'playlist' ? trim(payload.id) || sourcePlaylistId : sourcePlaylistId;
  const entries = (payload.entries || []).map((entry, index) => normalizeEntry(entry, playlistId, index)).filter((entry): entry is YoutubeResolvedEntry => Boolean(entry));
  const activeVideoId = entries[0]?.videoId || trim(payload.id) || (payload.webpage_url ? extractYouTubeVideoId(payload.webpage_url) : null) || sourceVideoId;
  const kind = payload._type === 'playlist' || entries.length > 0 || Boolean(playlistId && !activeVideoId) ? 'playlist' : 'video';
  return {
    kind,
    sourceUrl,
    title: trim(payload.title) || (kind === 'playlist' ? 'YouTube playlist' : 'YouTube video'),
    uploader: trim(payload.uploader) || trim(payload.channel),
    thumbnailUrl: trim(payload.thumbnail) || entries[0]?.thumbnailUrl || (activeVideoId ? thumbnail(activeVideoId) : null),
    embedUrl: embedUrl(activeVideoId, kind === 'playlist' ? playlistId : null),
    videoId: activeVideoId,
    playlistId,
    entries,
    resolvedWith: 'youtube-dl',
    warning: null,
  };
}

async function resolveWithOEmbed(sourceUrl: string): Promise<YoutubeResolvedMedia> {
  const videoId = extractYouTubeVideoId(sourceUrl);
  if (!videoId) throw new Error('A playable video ID could not be extracted from this YouTube URL.');
  const playlistId = extractYouTubePlaylistId(sourceUrl);
  let title = 'YouTube video';
  let uploader: string | null = null;
  let thumb: string | null = thumbnail(videoId);
  try {
    const res = await fetch(`https://www.youtube.com/oembed?format=json&url=${encodeURIComponent(sourceUrl)}`, { cache: 'no-store' });
    if (res.ok) {
      const json = (await res.json()) as { title?: string; author_name?: string; thumbnail_url?: string };
      title = trim(json.title) || title;
      uploader = trim(json.author_name);
      thumb = trim(json.thumbnail_url) || thumb;
    }
  } catch {
    // The embed still works without metadata.
  }
  return {
    kind: 'video',
    sourceUrl,
    title,
    uploader,
    thumbnailUrl: thumb,
    embedUrl: embedUrl(videoId, playlistId),
    videoId,
    playlistId,
    entries: [],
    resolvedWith: 'oembed',
    warning: playlistId ? 'Playlist entries need yt-dlp or youtube-dl on the server; playing the first video for now.' : null,
  };
}

export async function resolveYoutubeMedia(sourceUrl: string): Promise<YoutubeResolvedMedia> {
  const url = sourceUrl.trim();
  if (!isYouTubeUrl(url)) throw new Error('Paste a valid YouTube video or playlist URL.');

  const { execFile } = await import('node:child_process');
  const { promisify } = await import('node:util');
  const execFileAsync = promisify(execFile);

  for (const runner of runners) {
    try {
      const { stdout } = await execFileAsync(runner, ['--dump-single-json', '--flat-playlist', '--skip-download', '--no-warnings', '--playlist-end', '25', url], {
        timeout: 20_000,
        maxBuffer: 4 * 1024 * 1024,
      });
      return normalize(url, JSON.parse(stdout) as YoutubeDlPayload);
    } catch {
      // Try the next resolver, then oEmbed.
    }
  }
  return resolveWithOEmbed(url);
}
