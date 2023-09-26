// Copyright 2023 Vivek Panyam
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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