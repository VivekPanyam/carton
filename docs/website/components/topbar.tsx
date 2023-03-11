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