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