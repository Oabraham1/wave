// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
	site: 'https://wave.ojima.me',
	integrations: [
		starlight({
			title: 'WAVE',
			description: 'The Universal GPU ISA - Write once, run on any GPU',
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/Oabraham1/wave' },
			],
			customCss: ['./src/styles/custom.css'],
			sidebar: [
				{
					label: 'Getting Started',
					items: [
						{ label: 'Introduction', slug: 'getting-started/introduction' },
						{ label: 'Installation', slug: 'getting-started/installation' },
						{ label: 'Quick Start', slug: 'getting-started/quickstart' },
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
						{ label: 'Writing Kernels', slug: 'guides/writing-kernels' },
						{ label: 'Memory Model', slug: 'guides/memory-model' },
						{ label: 'Control Flow', slug: 'guides/control-flow' },
						{ label: 'Optimization', slug: 'guides/optimization' },
					],
				},
				{
					label: 'Reference',
					items: [
						{ label: 'WAVE Spec', slug: 'reference/spec' },
					{ label: 'Spec v0.3', slug: 'reference/spec-v03' },
					{ label: 'Spec v0.2', slug: 'reference/spec-v02' },
					{ label: 'Spec v0.1', slug: 'reference/spec-v01' },
						{ label: 'Python API', slug: 'reference/python-api' },
						{ label: 'Rust API', slug: 'reference/rust-api' },
						{ label: 'C/C++ API', slug: 'reference/cpp-api' },
						{ label: 'TypeScript API', slug: 'reference/typescript-api' },
						{ label: 'Instruction Set', slug: 'reference/instruction-set' },
						{ label: 'CLI Tools', slug: 'reference/cli-tools' },
					],
				},
				{
					label: 'Architecture',
					items: [
						{ label: 'Overview', slug: 'architecture/overview' },
						{ label: 'Compiler', slug: 'architecture/compiler' },
						{ label: 'Backends', slug: 'architecture/backends' },
						{ label: 'Emulator', slug: 'architecture/emulator' },
						{ label: 'ISA Design', slug: 'architecture/isa-design' },
					],
				},
				{
					label: 'Internals',
					collapsed: true,
					items: [
						{ label: 'Binary Encoding', slug: 'internals/binary-encoding' },
						{ label: 'Register Model', slug: 'internals/register-model' },
						{ label: 'Control Flow', slug: 'internals/control-flow' },
						{ label: 'Memory Scoping', slug: 'internals/memory-scoping' },
						{ label: 'Modifier Field Evolution', slug: 'internals/modifier-field' },
						{ label: 'Occupancy Equation', slug: 'internals/occupancy-equation' },
						{ label: 'Spec Defects Found', slug: 'internals/spec-defects' },
						{ label: 'Cross-Vendor Analysis', slug: 'internals/cross-vendor-analysis' },
						{ label: 'The Shuffle Primitive', slug: 'internals/shuffle-primitive' },
						{ label: 'Thin Abstraction Principle', slug: 'internals/thin-abstraction' },
						{ label: 'Backend Mapping', slug: 'internals/backend-mapping' },
					],
				},
				{
					label: 'Research',
					items: [
						{ label: 'Papers', slug: 'research/papers' },
						{ label: 'Hardware Verification', slug: 'research/hardware-verification' },
						{ label: 'Contributing', slug: 'research/contributing' },
					],
				},
			],
		}),
	],
});
