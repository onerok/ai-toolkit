import { NextRequest, NextResponse } from 'next/server';
import OpenAI from 'openai';
import fs from 'fs';
import path from 'path';
import yaml from 'js-yaml';

// Default prompt matching the local VLM default
const DEFAULT_PROMPT = `caption this image. describe every single thing in the image in detail. Do not include any unnecessary words in your description for the sake of good grammar. I want many short statements that serve the single purpose of giving the most thorough description if items as possible in the smallest, comma separated way possible. be sure to describe people's moods, clothing, the environment, lighting, colors, and everything.`;

interface CaptionConfig {
  prompt: string;
}

function loadDatasetConfig(imagePath: string): CaptionConfig {
  const config: CaptionConfig = {
    prompt: DEFAULT_PROMPT,
  };

  // Look for .caption_config.yaml in the image's directory
  const imageDir = path.dirname(imagePath);
  const configPath = path.join(imageDir, '.caption_config.yaml');

  if (fs.existsSync(configPath)) {
    try {
      const fileContent = fs.readFileSync(configPath, 'utf-8');
      const yamlConfig = yaml.load(fileContent) as { caption?: { prompt?: string } };

      if (yamlConfig?.caption?.prompt) {
        config.prompt = yamlConfig.caption.prompt;
        console.log(`[OpenAI Caption] Loaded dataset config from ${configPath}`);
      }
    } catch (e) {
      console.error(`[OpenAI Caption] Error loading config from ${configPath}:`, e);
    }
  }

  return config;
}

function getImageMediaType(imagePath: string): string {
  const ext = path.extname(imagePath).toLowerCase();
  const mediaTypes: Record<string, string> = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.gif': 'image/gif',
    '.webp': 'image/webp',
  };
  return mediaTypes[ext] || 'image/jpeg';
}

export async function POST(request: NextRequest) {
  const body = await request.json();
  const { imagePath, prompt: customPrompt } = body;

  if (!imagePath) {
    return NextResponse.json({ error: 'imagePath is required' }, { status: 400 });
  }

  // Check for API key
  const apiKey = process.env.OPENAI_API_KEY;
  console.log('[OpenAI Caption] API key present:', !!apiKey, apiKey ? `(${apiKey.slice(0, 7)}...)` : '');

  if (!apiKey) {
    return NextResponse.json(
      { error: 'OPENAI_API_KEY environment variable not set. Add it to ui/.env.local' },
      { status: 500 }
    );
  }

  // Verify image exists
  if (!fs.existsSync(imagePath)) {
    return NextResponse.json({ error: `Image not found: ${imagePath}` }, { status: 404 });
  }

  // Load dataset config
  const config = loadDatasetConfig(imagePath);
  const prompt = customPrompt || config.prompt;
  console.log(`[OpenAI Caption] Using prompt: "${prompt.slice(0, 80)}..."`);

  try {
    // Read and encode image as base64
    const imageBuffer = fs.readFileSync(imagePath);
    const base64Image = imageBuffer.toString('base64');
    const mediaType = getImageMediaType(imagePath);

    // Initialize OpenAI client
    const openai = new OpenAI({ apiKey });

    // Call OpenAI Responses API (newer format for GPT-5 models)
    const response = await openai.responses.create({
      model: 'gpt-5-mini',
      input: [
        {
          role: 'user',
          content: [
            { type: 'input_text', text: prompt },
            {
              type: 'input_image',
              image_url: `data:${mediaType};base64,${base64Image}`,
            },
          ],
        },
      ],
    });

    // Debug: log the full response structure
    console.log('[OpenAI Caption] Full response:', JSON.stringify(response, null, 2));

    // Extract caption from response
    const caption = response.output_text?.trim() || '';
    console.log(`[OpenAI Caption] Extracted caption: "${caption}"`);

    // Save caption to file (same behavior as local VLM)
    const captionPath = imagePath.replace(/\.[^/.]+$/, '.txt');
    fs.writeFileSync(captionPath, caption);
    console.log(`[OpenAI Caption] Saved caption to ${captionPath}`);

    return NextResponse.json({
      caption,
      provider: 'openai',
      model: 'gpt-5-mini',
    });
  } catch (error: any) {
    console.error('[OpenAI Caption] Error:', error);

    if (error?.status === 401) {
      return NextResponse.json({ error: 'Invalid OpenAI API key' }, { status: 401 });
    }

    return NextResponse.json(
      { error: error?.message || 'Failed to generate caption' },
      { status: 500 }
    );
  }
}
