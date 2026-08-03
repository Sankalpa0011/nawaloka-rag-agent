import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Nawaloka Hospitals Assistant",
  description:
    "Hybrid agent routing patient questions between the hospital database and nawaloka.com.",
  icons: { icon: "/favicon.svg" },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
