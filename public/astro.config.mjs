import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import starlight from '@astrojs/starlight';
import starlightThemeBlack from 'starlight-theme-black';

export default defineConfig({
	site: 'https://openguard.sh',
	integrations: [
		sitemap(),
		starlight({
			title: 'OpenGuard',
			disable404Route: true,
			customCss: ['./src/styles/custom.css'],
			plugins: [
				starlightThemeBlack({
					navLinks: [
						{ label: 'Home', link: '/' },
					],
					footerText: 'Built with [Astro Starlight](https://starlight.astro.build).',
				}),
			],
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/Jitera-Labs/openguard' }],
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
