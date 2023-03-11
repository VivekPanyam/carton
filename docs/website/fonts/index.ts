import { IBM_Plex_Sans, Roboto_Mono } from '@next/font/google'
import LocalFont from '@next/font/local'

export const plex_sans = IBM_Plex_Sans({ weight: ["200", "300", "400", "500", "600", "700"], subsets: ["latin"], display: "block" })
export const roboto_mono = Roboto_Mono({ weight: "400", subsets: ["latin"], display: "block" })
export const virgil = LocalFont({ src: './Virgil.woff2', display: "block" })