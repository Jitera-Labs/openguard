import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightThemeBlack from 'starlight-theme-black';

export default defineConfig({
	site: 'https://openguard.sh',
	integrations: [
		starlight({
			title: 'OpenGuard',
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
