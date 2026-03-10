import { defineCollection, z } from 'astro:content';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';
import { glob } from 'astro/loaders';

export const collections = {
	docs: defineCollection({ loader: docsLoader(), schema: docsSchema() }),
	blog: defineCollection({
		loader: glob({ pattern: '*.mdx', base: './src/content/blog' }),
		schema: z.object({
			title: z.string(),
			description: z.string().min(50).max(160),
			date: z.coerce.date(),
			author: z.string().optional().default('OpenGuard Team'),
			tags: z.array(z.string()).optional(),
		}),
	}),
};
