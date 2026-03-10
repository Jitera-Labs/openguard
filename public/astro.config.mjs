import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import starlight from '@astrojs/starlight';
import starlightThemeBlack from 'starlight-theme-black';
import { docsHeaderNavLinks, githubCtaAriaLabel, githubUrl } from './src/config/site-header.mjs';

export default defineConfig({
	site: 'https://openguard.sh',
	vite: { cacheDir: '/tmp/vite-cache-openguard' },
	integrations: [
		sitemap(),
		starlight({
			title: 'OpenGuard',
			disable404Route: true,
			customCss: ['./src/styles/custom.css'],
			components: {
				Header: './src/components/starlight/Header.astro',
			},
			plugins: [
				starlightThemeBlack({
					navLinks: docsHeaderNavLinks,
					footerText: 'Built with [Astro Starlight](https://starlight.astro.build).',
				}),
			],
			social: [{ icon: 'github', label: githubCtaAriaLabel, href: githubUrl }],
			sidebar: [
				{ label: 'Getting Started', slug: 'docs/getting-started' },
				{ label: 'Configuration', slug: 'docs/configuration' },
				{
					label: 'Guards',
					autogenerate: { directory: 'docs/guards' },
				},
			],
		}),
	],
});
