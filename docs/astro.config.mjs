// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
	site: 'https://wave.ojima.me',
	markdown: { smartypants: false },
	integrations: [
		starlight({
			title: 'WAVE',
			description: 'A universal GPU instruction set architecture',
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/Oabraham1/wave' },
			],
			customCss: ['./src/styles/custom.css'],
			sidebar: [
				{
					label: 'Getting started',
					items: [
						{ label: 'Introduction', slug: 'getting-started/introduction' },
						{ label: 'Installation', slug: 'getting-started/installation' },
						{ label: 'Quick start', slug: 'getting-started/quickstart' },
						{ label: 'Supported GPUs', slug: 'getting-started/supported-gpus' },
					],
				},
				{
					label: 'Guides',
					items: [
						{ label: 'Python SDK', slug: 'guides/python-sdk' },
						{ label: 'Rust SDK', slug: 'guides/rust-sdk' },
						{ label: 'C/C++ SDK', slug: 'guides/cpp-sdk' },
						{ label: 'TypeScript SDK', slug: 'guides/typescript-sdk' },
						{ label: 'Writing kernels', slug: 'guides/writing-kernels' },
						{ label: 'Memory model', slug: 'guides/memory-model' },
						{ label: 'Control flow', slug: 'guides/control-flow' },
						{ label: 'Optimization', slug: 'guides/optimization' },
					],
				},
				{
					label: 'Reference',
					items: [
						{ label: 'WAVE specification', slug: 'reference/spec' },
						{ label: 'Specification v0.4', slug: 'reference/spec-v04' },
						{ label: 'Specification v0.3', slug: 'reference/spec-v03' },
						{ label: 'Specification v0.2', slug: 'reference/spec-v02' },
						{ label: 'Specification v0.1', slug: 'reference/spec-v01' },
						{ label: 'Python API', slug: 'reference/python-api' },
						{ label: 'Rust API', slug: 'reference/rust-api' },
						{ label: 'C/C++ API', slug: 'reference/cpp-api' },
						{ label: 'TypeScript API', slug: 'reference/typescript-api' },
						{ label: 'Instruction set', slug: 'reference/instruction-set' },
						{ label: 'CLI tools', slug: 'reference/cli-tools' },
					],
				},
				{
					label: 'Architecture',
					items: [
						{ label: 'Overview', slug: 'architecture/overview' },
						{ label: 'Compiler', slug: 'architecture/compiler' },
						{ label: 'Backends', slug: 'architecture/backends' },
						{ label: 'Emulator', slug: 'architecture/emulator' },
						{ label: 'ISA design', slug: 'architecture/isa-design' },
					],
				},
				{
					label: 'Internals',
					collapsed: true,
					items: [
						{ label: 'Binary encoding', slug: 'internals/binary-encoding' },
						{ label: 'Register model', slug: 'internals/register-model' },
						{ label: 'Control flow', slug: 'internals/control-flow' },
						{ label: 'Memory scoping', slug: 'internals/memory-scoping' },
						{ label: 'Modifier field', slug: 'internals/modifier-field' },
						{ label: 'Occupancy equation', slug: 'internals/occupancy-equation' },
						{ label: 'Encoding and execution invariants', slug: 'internals/spec-defects' },
						{ label: 'Cross-vendor analysis', slug: 'internals/cross-vendor-analysis' },
						{ label: 'The shuffle primitive', slug: 'internals/shuffle-primitive' },
						{ label: 'Thin abstraction principle', slug: 'internals/thin-abstraction' },
						{ label: 'Backend mapping', slug: 'internals/backend-mapping' },
					],
				},
				{
					label: 'Research',
					items: [
						{ label: 'Papers', slug: 'research/papers' },
						{ label: 'Hardware verification', slug: 'research/hardware-verification' },
						{ label: 'Contributing', slug: 'research/contributing' },
					],
				},
			],
		}),
	],
});
