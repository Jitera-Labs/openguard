import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightThemeBlack from 'starlight-theme-black';

export default defineConfig({
	site: 'https://openguard.sh',
	integrations: [
		starlight({
			title: 'OpenGuard',
			disable404Route: true,
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
				{
					label: 'Guards',
					autogenerate: { directory: 'guards' },
				},
			],
		}),
	],
});
