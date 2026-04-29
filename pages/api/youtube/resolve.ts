import type { NextApiRequest, NextApiResponse } from 'next';
import { isYouTubeUrl, resolveYoutubeMedia } from '@/lib/youtube';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    res.setHeader('allow', 'POST');
    return res.status(405).json({ error: 'Method not allowed.' });
  }

  const url = typeof req.body?.url === 'string' ? req.body.url.trim() : '';
  if (!url || !isYouTubeUrl(url)) {
    return res.status(400).json({ error: 'Paste a valid YouTube video or playlist URL.' });
  }

  try {
    const media = await resolveYoutubeMedia(url);
    return res.status(200).json(media);
  } catch (error) {
    return res.status(502).json({
      error: error instanceof Error ? error.message : 'YouTube media could not be resolved right now.',
    });
  }
}
