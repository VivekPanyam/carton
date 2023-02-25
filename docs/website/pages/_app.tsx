import '@/styles/globals.css'
import '@/styles/prism_style.css'
import type { AppProps } from 'next/app'

import { LanguageContext, DEFAULT_LANGUAGE } from '@/components/languageselect'
import { useState } from 'react'
import { MDXProvider } from '@mdx-js/react'
import Code from '@/components/code'
import Link from 'next/link'

const Pre = ({ children, className, ...props }: any) => {
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

  return <Code language={language} codeString={children.props.children.trimEnd()} className={`not-prose -ml-5 md:ml-0 ${className || ""}`} {...props} />
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
  const [currentLanguage, setCurrentLanguage] = useState(DEFAULT_LANGUAGE)
  return (
    <MDXProvider components={components}>
      <LanguageContext.Provider value={{
        currentLanguage,
        setCurrentLanguage
      }}>
        <Component {...pageProps} />
      </LanguageContext.Provider>
    </MDXProvider>
  )
}
