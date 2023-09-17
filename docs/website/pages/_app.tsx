import '@/styles/globals.css'
import '@/styles/prism_style.css'
import type { AppProps } from 'next/app'

import { LanguageContext, DEFAULT_LANGUAGE, LANGUAGES } from '@/components/languageselect'
import { useContext, useEffect, useState } from 'react'
import { MDXProvider } from '@mdx-js/react'
import Code from '@/components/code'
import Link from 'next/link'

const Pre = ({ children, className, forLang, ...props }: any) => {
  const { currentLanguage, setCurrentLanguage: _ } = useContext(LanguageContext)

  if (children.type != "code") {
    throw `Expected child type to be \`code\`, but got ${children.type}`
  }

  if (children.props.className == undefined) {
    return <></>
  }

  let language: string = children.props.className
  if (!language.startsWith("language-")) {
    throw `Expected classname to start with \`language-\`, but got ${children.props.className}`
  }

  language = language.replace("language-", "")

  // If forLang is set, we only want to display this block for a specific language
  if (forLang != null) {
    const target = forLang.split(',')
    const currLang = currentLanguage.name.toLowerCase();
    if (target.indexOf(currLang) < 0) {
      return null
    }
  }

  return <Code language={language} codeString={children.props.children.trimEnd()} className={`not-prose -ml-5 overflow-x-auto -mr-5 md:ml-0 md:mr-0 ${className || ""}`} {...props} />
}

// See https://github.com/vercel/next.js/discussions/11110
const CustomLink = (props: any) => {
  const href = props.href;

  if (href.startsWith('/')) {
    return (
      <Link href={href} {...props}>
        {props.children}
      </Link>
    );
  }

  if (href.startsWith('#')) {
    return <a {...props} />;
  }

  return <a target="_blank" rel="noopener noreferrer" {...props} />;
};

const components = {
  pre: Pre,
  a: CustomLink,
}

export default function App({ Component, pageProps }: AppProps) {
  const [currentLanguage, setCurrentLanguage] = useState<typeof DEFAULT_LANGUAGE | null>(null)

  // Update the current language based on localStorage
  useEffect(() => {
    // Checks if we have a selectedLanguage in local storage and loads it
    const listener = () => {
      const storedLanguage = localStorage.getItem("selectedLanguage");
      if (storedLanguage) {
        const matching = LANGUAGES.filter((item) => item.name == storedLanguage);
        if (matching.length > 0) {
          setCurrentLanguage(matching[0])
          return
        }
      }

      // Fallback to default
      setCurrentLanguage(DEFAULT_LANGUAGE)
    };

    // Check if we have a stored language
    listener()

    // Disable sync for now because it could get annoying to use
    // // Update on changes
    // window.addEventListener("storage", listener);

    // // Remove the event listener when the component unmounts
    // return () => window.removeEventListener("storage", listener)
  }, [])

  // Save the current language
  useEffect(() => {
    if (currentLanguage != null) {
      localStorage.setItem("selectedLanguage", currentLanguage.name)
    }
  }, [currentLanguage])

  // If we don't have a language selected yet
  const cl = currentLanguage || DEFAULT_LANGUAGE;

  return (
    <MDXProvider components={components}>
      <LanguageContext.Provider value={{
        currentLanguage: cl,
        setCurrentLanguage
      }}>
        <Component {...pageProps} />
      </LanguageContext.Provider>
    </MDXProvider>
  )
}
