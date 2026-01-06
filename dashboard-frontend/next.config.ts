import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Enable compression
  compress: true,
  
  // Optimize images
  images: {
    formats: ['image/avif', 'image/webp'],
  },
  
  // Experimental features for better performance
  experimental: {
    optimizePackageImports: ['lucide-react', 'recharts'],
  },
  
  // Enable standalone output for Docker
  output: 'standalone',
  
  // Turbopack configuration (Next.js 16+ default)
  turbopack: {
    // Turbopack handles optimizations automatically
    // Package imports optimization is handled via experimental.optimizePackageImports
  },
  
  // Webpack optimizations (fallback for --webpack flag)
  webpack: (config, { isServer }) => {
    if (!isServer) {
      // Optimize bundle splitting
      config.optimization = {
        ...config.optimization,
        splitChunks: {
          chunks: 'all',
          cacheGroups: {
            default: false,
            vendors: false,
            // Vendor chunk for large libraries
            vendor: {
              name: 'vendor',
              chunks: 'all',
              test: /node_modules/,
              priority: 20,
            },
            // Separate chunk for recharts (large library)
            recharts: {
              name: 'recharts',
              chunks: 'all',
              test: /[\\/]node_modules[\\/]recharts[\\/]/,
              priority: 30,
            },
            // Common chunk for shared code
            common: {
              name: 'common',
              minChunks: 2,
              chunks: 'all',
              priority: 10,
              reuseExistingChunk: true,
            },
          },
        },
      };
    }
    return config;
  },
};

export default nextConfig;
