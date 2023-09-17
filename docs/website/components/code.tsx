import * as React from "react"
import Highlight, { defaultProps, Language } from "prism-react-renderer"
import vsDark from "prism-react-renderer/themes/vsDark"

// Add rust support
// @ts-ignore
import Prism from "prism-react-renderer/prism";

(typeof global !== "undefined" ? global : window).Prism = Prism;


require("prismjs/components/prism-rust")

// From https://github.com/LekoArts/gatsby-themes/blob/main/packages/themes-utils/src/index.ts
// (MIT licensed)
const calculateLinesToHighlight = (meta: string) => {
    if (!meta) {
        return () => false
    }
    const lineNumbers = meta.split(`,`).map((v) => v.split(`-`).map((x) => parseInt(x, 10)))
    return (index: number) => {
        const lineNumber = index + 1
        return lineNumbers.some(([start, end]) => (end ? lineNumber >= start && lineNumber <= end : lineNumber === start))
    }
}



type CodeProps = {
    codeString: string
    withLineNumbers?: boolean
    showTag?: boolean,
    highlight?: string
    language: "python" | "bash" // | "rust", // Language
    className?: string,
    linePrompt?: string,
}

// Based on https://github.com/LekoArts/gatsby-themes/blob/main/themes/gatsby-theme-minimal-blog/src/components/code.tsx
// (MIT licensed)

const LANGUAGE_TAGS = {
    "python": { "bg": "bg-sky-500", tag: "py"},
    "bash": {"bg": "bg-amber-400", tag: "bash"},
    "text": {"bg": "", tag: ""},
    "js": {"bg": "bg-emerald-500", tag: "js"},
    "rust": {"bg": "bg-violet-500", tag: "rust"},
}

const Code = ({
    codeString,
    withLineNumbers = false,
    showTag = true,
    className: wrapperClassName = "",
    language,
    highlight = ``,
    linePrompt,
}: CodeProps) => {
    const shouldHighlightLine = calculateLinesToHighlight(highlight)
    const shouldShowLineNumbers = withLineNumbers

    const languageTag = LANGUAGE_TAGS[language]

    return (
        <Highlight
            {...defaultProps}
            code={codeString}
            // @ts-ignore
            language={language}
            theme={vsDark}
        >
            {({ className, tokens, getLineProps, getTokenProps }) => (
                <React.Fragment>
                    <div className={`relative text-sm sm:text-base ${wrapperClassName}`} data-language={language}>
                        {showTag && <div className={`rounded-b ${languageTag.bg} text-white uppercase tracking-wide py-0.5 px-2 absolute text-right font-bold left-4`}>{languageTag.tag}</div> }
                        <pre className={`float-left min-w-full ${className}`} data-linenumber={shouldShowLineNumbers}>
                            <code className={`p-4 float-left min-w-full language-${language} ${showTag && "pt-8"}`}>
                                {tokens.map((line, i) => {
                                    const lineProps = getLineProps({ line, key: i })
                                    const additionalLineNumberProps: React.CSSProperties = {}
                                    let additionalLineNumberClasses = "";

                                    if (shouldHighlightLine(i)) {
                                        lineProps.className = `${lineProps.className} border-l-4 border-l-sky-400 bg-slate-500 bg-opacity-10 -ml-[4px]`
                                        additionalLineNumberProps.opacity = 0.5;
                                    }

                                    return (
                                        <div key={i} {...lineProps}>
                                            {shouldShowLineNumbers && <span className={`inline-block w-12 select-none opacity-30 text-center relative ${additionalLineNumberClasses}`} style={additionalLineNumberProps}>{i + 1}</span>}
                                            {linePrompt != undefined && <span className={`inline-block w-12 select-none opacity-30 text-center relative ${additionalLineNumberClasses}`} style={additionalLineNumberProps}>{linePrompt}</span>}
                                            {line.map((token, key) => (
                                                <span key={key} {...getTokenProps({ token, key })} />
                                            ))}
                                        </div>
                                    )
                                })}
                            </code>
                        </pre>
                        <div className="clear-both"/>
                    </div>
                </React.Fragment>
            )}
        </Highlight>
    )
}

export default Code