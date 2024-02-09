/** @type {import('next').NextConfig} */
// const nextConfig = {}

// module.exports = nextConfig

module.exports = () => {
    const rewrites = () => {
        return [
            {
                source: "/hello/:path*",
                destination: "http://52.15.65.69:5000/hello/:path*",
            },
        ];
    };
    return {
        rewrites,
    };
};