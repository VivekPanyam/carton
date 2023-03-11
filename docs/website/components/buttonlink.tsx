import Link from "next/link"

export function ButtonLink({ children, href }: any) {
    return (
        <Link href={href}>
            <div className="px-8 py-4 text-center rounded-md border flex cursor-pointer select-none hover:bg-slate-100">
                {children}
            </div>
        </Link>
    )
}

export function ButtonGrid({ children }: any) {
    return (
        <div className="flex flex-wrap gap-4 not-prose">
            {children}
        </div>
    )
}