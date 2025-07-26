/** @type {import('next').NextConfig} */
const nextConfig = {
  transpilePackages: [
    '@segment3d/ui',
    '@segment3d/config', 
    '@segment3d/types',
    '@segment3d/utils'
  ],
  experimental: {
    appDir: true,
  },
  webpack: (config) => {
    // Handle Three.js and other 3D libraries
    config.externals = config.externals || {};
    config.externals['three'] = 'three';
    
    return config;
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/:path*',
      },
    ];
  },
};

module.exports = nextConfig; 