export const docsPath = '/docs/getting-started/';

export const siteHeaderNavLinks = [
	{ label: 'Home', href: '/' },
	{ label: 'Docs', href: docsPath },
	{ label: 'Blog', href: '/blog/' },
];

export const docsHeaderNavLinks = siteHeaderNavLinks.map(({ label, href }) => ({
	label,
	link: href,
}));

export const githubUrl = 'https://github.com/Jitera-Labs/openguard';
export const githubCtaLabel = 'Star';
export const githubCtaAriaLabel = 'Star OpenGuard on GitHub';