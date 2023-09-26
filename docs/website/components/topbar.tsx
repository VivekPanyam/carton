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

import Link from "next/link"
import { SiGithub } from "react-icons/si"
import LanguageSelect from "./languageselect"

export default function TopBar({showLanguageSelector = true}) {
    return (
        <div className="py-5 flex text-sm sm:text-base justify-between items-center">
            <Link href="/"><h1 className="text-2xl">Carton</h1></Link>
            <div className='flex space-x-3 sm:space-x-5 items-center'>
                <Link href="/quickstart"><span>Quickstart</span></Link>
                <Link href="/docs"><span>Docs</span></Link>
                <Link target="_blank" href="https://carton.pub"><span>Models</span></Link>
                {showLanguageSelector && <LanguageSelect className="hidden sm:inline-block"  /> }
                <Link target="_blank" href="https://github.com/VivekPanyam/carton"><SiGithub className="text-base sm:text-lg" /></Link>
            </div>
        </div>
    )
}