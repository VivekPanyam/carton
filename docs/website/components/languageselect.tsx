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

import { createContext, Fragment, useContext, useState } from 'react'
import { Listbox, Transition } from '@headlessui/react'
import { HiCheck, HiChevronUpDown } from 'react-icons/hi2'
import React from 'react';

export const LANGUAGES = [
    { name: "Python", enabled: true },
    { name: "JavaScript", enabled: true },
    { name: "TypeScript", enabled: true },
    { name: "Rust", enabled: true },
    { name: "C", enabled: true },
    { name: "C++", enabled: true },
    { name: "C#", enabled: false },
    { name: "Java", enabled: false },
    { name: "Golang", enabled: false },
    { name: "Swift", enabled: false },
    { name: "Ruby", enabled: false },
    { name: "PHP", enabled: false },
    { name: "Kotlin", enabled: false },
    { name: "Scala", enabled: false },
]

export const DEFAULT_LANGUAGE = LANGUAGES[0];
export const LanguageContext = createContext<{ currentLanguage: any, setCurrentLanguage: any }>({ currentLanguage: DEFAULT_LANGUAGE, setCurrentLanguage: () => { } });

export default function LanguageSelect({ className }: any) {
    const { currentLanguage, setCurrentLanguage } = useContext(LanguageContext)

    return (
        <div className={className || ""}>
            <Listbox value={currentLanguage} onChange={setCurrentLanguage}>
                <div className="relative">
                    <Listbox.Button className="relative border cursor-pointer rounded-lg bg-white py-2 pl-3 pr-10 text-left shadow-sm focus:outline-none focus-visible:border-indigo-500 focus-visible:ring-2 focus-visible:ring-white focus-visible:ring-opacity-75 focus-visible:ring-offset-2 focus-visible:ring-offset-orange-300 sm:text-sm">
                        <span className="block truncate">{currentLanguage.name}</span>
                        <span className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-2">
                            <HiChevronUpDown
                                className="h-5 w-5 text-gray-400"
                                aria-hidden="true"
                            />
                        </span>
                    </Listbox.Button>
                    <Transition
                        as={Fragment}
                        leave="transition ease-in duration-100"
                        leaveFrom="opacity-100"
                        leaveTo="opacity-0"
                    >
                        <Listbox.Options className="absolute right-0 mt-1 max-h-60 w-36 overflow-auto rounded-md bg-white py-1 text-base shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none sm:text-sm z-50">
                            {LANGUAGES.map((lang, langIdx) => (
                                <Listbox.Option
                                    key={langIdx}
                                    className={({ active, disabled }) =>
                                        `relative cursor-default select-none py-2 pl-10 pr-4 ${active ? 'bg-amber-100 text-amber-900' : 'text-gray-900'} ${disabled && 'text-slate-400 bg-slate-200'}`
                                    }
                                    disabled={!lang.enabled}
                                    value={lang}
                                >
                                    {({ selected, disabled }) => (
                                        <>
                                            <span
                                                title={disabled ? 'This language is not yet supported. Check back soon!' : undefined}
                                                className={`block truncate ${selected ? 'font-medium' : 'font-normal'} ${!disabled && 'cursor-pointer'}`}
                                            >
                                                {lang.name}
                                            </span>
                                            {selected ? (
                                                <span className="absolute inset-y-0 left-0 flex items-center pl-3 text-amber-600">
                                                    <HiCheck className="h-5 w-5" aria-hidden="true" />
                                                </span>
                                            ) : null}
                                        </>
                                    )}
                                </Listbox.Option>
                            ))}
                        </Listbox.Options>
                    </Transition>
                </div>
            </Listbox>
        </div>
    )
}

/// Show at most one child based on the selected language
export const LanguageSwitch = ({ children }: any) => {
    const { currentLanguage, setCurrentLanguage: _ } = useContext(LanguageContext)
    const arrayChildren = React.Children.toArray(children);

    let selectedChild = null;
    for (const child of arrayChildren) {
        // @ts-ignore
        const forLang = child.props.forLang;

        // If forLang is set, we only want to display this block for a specific language
        if (forLang != null) {
            const target = forLang.split(',')
            const currLang = currentLanguage.name.toLowerCase();
            if (target.indexOf(currLang) >= 0) {
                selectedChild = child;
                break;
            }
        } else {
            // This is a catch all
            selectedChild = child;
            break;
        }
    }

    return (
        <>
            {selectedChild}
        </>
    )
}

export const LanguageItem = ({ children, forLang}: any) => {
    return children
}