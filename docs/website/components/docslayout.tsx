import TopBar from "@/components/topbar";
import { plex_sans } from "@/fonts";
import Head from "next/head";

export default function DocsLayout({ children, className, title }: any) {
    return (
        <div className={plex_sans.className + ' overflow-hidden bg-slate-50 min-h-screen pb-20'}>
            <Head>
                <title>{title || "Docs"} | Carton</title>
            </Head>
            <div className="drop-shadow-sm bg-white mb-5 lg:mb-20">
                <div className="max-w-6xl mx-5 xl:m-auto">
                    <TopBar />
                </div>
            </div>

            <div className={`max-w-6xl mx-5 xl:m-auto prose prose-sm sm:prose-base ${className || ""}`}>
                {children}
            </div>
        </div>)
}