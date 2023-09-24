import Head from 'next/head'
import Code from '../components/code'
import React, { useEffect, useState } from 'react'
import { IconContext } from "react-icons";

import { SiPytorch, SiTensorflow, SiKeras, SiOnnx, SiPython } from "react-icons/si";
import { AiOutlineFile, AiOutlineFolderOpen } from "react-icons/ai";
import { HiOutlinePlus, HiOutlineMinus } from "react-icons/hi";
import TopBar from '@/components/topbar'
import { plex_sans, roboto_mono, virgil } from '@/fonts';
import { HiArrowsUpDown, HiOutlineArrowDown } from 'react-icons/hi2';
import Link from 'next/link';

const LANGUAGES = [
  "Python",
  "JavaScript",
  "TypeScript",
  "Rust",
  "C",
  "C++",
  "C#",
  "Java",
  "Golang",
  "Swift",
  "Ruby",
  "PHP",
  "Kotlin",
  "Scala",
]

const FRAMEWORKS = [
  "PyTorch",
  "TorchScript",
  "TensorFlow",
  "JAX",
  "Keras",
  "TensorRT",
  "Ludwig",
  "Xgboost",
  "Lightgbm",
  "ONNX",
  "Caffe",
  "Huggingface transformers",
  "Huggingface diffusers",
  "Aribitrary python code",
]

// Used to pad the header code so it's always the same length
const MAX_FRAMEWORK_LEN = FRAMEWORKS.map(item => item.toLowerCase().replaceAll(" ", "_").length).reduce((a, b) => Math.max(a, b))

const get_header_code = (frameworkname: string) =>
  `import cartonml as carton

MODEL_PATH = "/path/to/${frameworkname}_model.carton"${" ".repeat(MAX_FRAMEWORK_LEN - frameworkname.length)}

model = await carton.load(MODEL_PATH)
await model.infer({
  "x": np.zeros(5)
})`

const RotatingLanguage = () => {
  let [index, setIndex] = useState(0)

  // increment index
  useEffect(() => {
    const interval = setInterval(() => {
      setIndex(idx => (idx + 1) % LANGUAGES.length)
    }, 1000)

    return () => clearInterval(interval)
  }, [])

  let lang = LANGUAGES[index]
  return <>{lang}</>
}

const RotatingFrameworkHeader = () => {
  let [index, setIndex] = useState(0)

  // increment index
  useEffect(() => {
    const interval = setInterval(() => {
      setIndex(idx => (idx + 1) % FRAMEWORKS.length)
    }, 1000)

    return () => clearInterval(interval)
  }, [])

  let framework = FRAMEWORKS[index].toLowerCase().replaceAll(" ", "_")

  return (
    <Code codeString={get_header_code(framework)} withLineNumbers={true} highlight="3" language="python" className="max-w-2xl overflow-x-auto m-auto" />
  )

}

export default function Home() {
  return (
    <div className={plex_sans.className + ' overflow-hidden'}>
      <Head>
        <title>Carton - Run any ML model from any programming language.</title>
      </Head>
      <div className="border-b bg-slate-50 pb-10">
        <div className="max-w-6xl mx-5 xl:m-auto">
          <div className='mb-10 lg:mb-20'>
            <TopBar showLanguageSelector={false} />
          </div>
          <h1 className="text-4xl sm:text-6xl font-extrabold text-center pb-10">
            Run any ML model from any programming language.
            <span className='text-slate-300 inline-block'> *</span>
          </h1>
          <p className="text-center max-w-3xl m-auto pb-10">
            One API for all frameworks.
          </p>
          <RotatingFrameworkHeader />
        </div>
        <div className="py-10 lg:py-20 left-0 max-w-6xl m-auto overflow-hidden relative">
          <div className="animate-[slide_60s_linear_infinite] absolute left-0">
            {
              // Display each language twice
              [1, 2].flatMap((idx) => LANGUAGES.map((lang) => <span key={`${lang}-${idx}`} className={`${roboto_mono.className} select-none uppercase px-20 text-slate-500`}>
                {lang}
                {["python", "rust"].indexOf(lang.toLowerCase()) == -1 && <span className='text-sky-400'>*</span>}
              </span>))
            }
          </div>
          <div className='bg-gradient-to-r from-slate-50 h-full w-20 left-0 top-0 absolute'></div>
          <div className='bg-gradient-to-l from-slate-50 h-full w-20 right-0 top-0 absolute'></div>
          <div className='invisible'>placeholder</div>
        </div>
        <div className="mx-5 sm:m-auto pb-10 max-w-3xl hidden sm:flex justify-between">
          <IconContext.Provider value={{ size: "50", className: "text-slate-500 inline-block flex-none" }}>
            <SiPytorch />
            <SiTensorflow />
            <SiKeras />
            <SiOnnx />
            <SiPython />
          </IconContext.Provider>
        </div>
        <div className={`text-center text-sky-400 ${virgil.className}`}>
          * Work in progress
        </div>
      </div>
      <div className='px-5'>
        {/* <div className='pt-10 lg:pt-20 -mb-10 lg:-mb-20 flex items-center justify-center'>
          <div className='text-center rounded-lg shadow-xl text-white px-10 py-5 bg-gradient-to-r from-blue-500 to-blue-600'>
            Want to run existing models? Check out the <Link target="_blank" href="https://carton.pub" className='underline'>community model registry.</Link>
          </div>
        </div> */}
        <div className='flex flex-col lg:flex-row max-w-6xl mx-auto py-20 lg:py-40 items-center'>
          <div className='flex-1 text-slate-800 mb-20 lg:mb-0'>
            <h1 className='relative text-4xl font-bold mb-5 text-center lg:text-left'><span className='text-slate-400 xl:absolute pr-5 -left-10 inline-block'>1 </span>Pack a model</h1>
            Carton wraps your model with some metadata and puts it in a zip file. It <i>does not</i> modify the original model, avoiding error-prone conversion steps.
            <br /><br />
            You just need to specify a framework and required version of that framework.
          </div>
          <div className='flex-1 flex flex-col justify-center items-center gap-y-4'>
            <div className='flex flex-col lg:flex-row justify-center items-center gap-y-4 lg:gap-x-4'>
              <div className='border w-64 h-20 bg-slate-100 flex justify-center items-center'>
                <span className={`${roboto_mono.className}`}>original_model.pt</span>
              </div>
              <span>requires</span>
              <span className={`${roboto_mono.className} text-emerald-500`}>TorchScript 2.0.x</span>
            </div>
            <HiOutlineArrowDown size={30} />
            <div className='ring-1 ring-slate-700/10 rounded-lg shadow-black/5 shadow-xl p-5 transition-all'>
              <AiOutlineFolderOpen className='text-xl text-slate-400 inline-block' /> <span className={`${roboto_mono.className} text-slate-400`}>model/</span><br />
              <AiOutlineFile className='text-xl text-slate-400 inline-block' /> <span className={`${roboto_mono.className} text-slate-700 ml-5`}>original_model.pt</span><br />
              <AiOutlineFile className='text-xl text-slate-400 inline-block' /> <span className={`${roboto_mono.className} text-slate-400`}>carton.toml</span><br />
              <span className={`${roboto_mono.className} text-slate-400`}>...</span><br />
            </div>
            <span className={`${roboto_mono.className} text-slate-400 text-xs -mt-2`}>model.carton</span>
          </div>
        </div>

        <div className='flex flex-col lg:flex-row max-w-6xl mx-auto py-20 lg:py-40 items-center border-t'>
          <div className='flex-1 text-slate-800 mb-20 lg:mb-0'>
            <h1 className='relative text-4xl font-bold mb-5 text-center lg:text-left'><span className='text-slate-400 xl:absolute pr-5 -left-10 inline-block'>2 </span>Load a model</h1>
            {`When loading a packed model, Carton reads the included metadata to figure out the appropriate "runner" to use and automatically fetches one if needed.`}
            <br />
            <br />
            <div className='text-sm text-slate-500'>Tip: a runner is a component of Carton that knows how to run a model with a specifc version of an ML framework.</div>
          </div>
          <div className='flex-1 flex flex-col justify-center items-center gap-y-4 mt-5 lg:mt-0'>
            <div className='ring-1 ring-slate-700/10 rounded-lg shadow-black/5 shadow-xl p-5 transition-all'>
              <span className={`${roboto_mono.className} text-slate-700`}>model.carton</span>
            </div>
            <HiOutlineArrowDown size={30} />
            <div className='flex flex-row justify-center items-center gap-x-1'>
              <span>Loading with</span>
              <span className={`text-emerald-500`}>TorchScript 2.0.1</span>
              <span>runner...</span>
            </div>
          </div>
        </div>


        <div className='flex flex-col lg:flex-row max-w-6xl mx-auto py-20 lg:py-40 items-center border-t'>
          <div className='flex-1 text-slate-800 mb-20 lg:mb-0'>
            <h1 className='relative text-4xl font-bold mb-5 text-center lg:text-left'><span className='text-slate-400 xl:absolute pr-5 -left-10 inline-block'>3 </span>Run a model</h1>
            All your inference code is framework-agnostic.
            <br /><br />
            {`Your application uses Carton's API and Carton calls into the underlying framework.`}
            <br /><br />
            Carton is implemented in Rust with bindings to several languages, all using the same optimized core.
          </div>
          <div className='flex-1 flex flex-col justify-center items-center gap-y-4'>
            <div className='border w-72 h-20 bg-slate-100 flex justify-center items-center'>
              <span className={`${roboto_mono.className}`}>Your <RotatingLanguage /> Application</span>
            </div>
            <HiArrowsUpDown size={30} />
            <div className='border w-64 h-20 bg-slate-100 flex justify-center items-center'>
              <span className={`${roboto_mono.className}`}>Carton</span>
            </div>
            <HiArrowsUpDown size={30} />
            <div className='border w-64 h-20 bg-slate-100 flex justify-center items-center'>
              <span className={`${roboto_mono.className}`}>TorchScript 2.0.1 Runner</span>
            </div>
          </div>
        </div>

        <div className='flex justify-center border-t py-20 items-center flex-col gap-10'>
          <Link href="/quickstart"><div className={`${roboto_mono.className} cursor-pointer shadow-xl rounded-md text-white py-5 px-20 text-xl bg-gradient-to-r from-pink-500 via-red-500 to-yellow-500`}>Get Started</div></Link>
          {/* <p>Get up and running in 5 minutes.</p> */}
          <p>or explore the <Link target="_blank" href="https://carton.pub" className='underline'>community model registry.</Link></p>
        </div>

        {/* (Embedded model explore component for "use an existing model")
        <div className='grid grid-cols-2'>
          <div className='flex flex-col items-center'>
            <h1 className="text-4xl font-extrabold text-center pt-20">
              Use an existing model
            </h1>
            <p className="text-center max-w-3xl m-auto py-5">
              Load it in one line of code
            </p>
            <Code codeString={`await carton.load("https://carton.pub/openai/gpt2")`} withLineNumbers={false} showTag={false} language="python" className="max-w-xl overflow-hidden m-auto" />
            <p className="text-center max-w-3xl m-auto py-5">
              Check out the community model registry.
            </p>
          </div>

          <div className='flex flex-col items-center'>
            <h1 className="text-4xl font-extrabold text-center pt-20">
              or create your own.
            </h1>
            <p className="text-center max-w-3xl m-auto py-5">
              That's easy too
            </p>
            <Code codeString={`await carton.pack("/path/to/orig_model", runner = "xgboost")`} withLineNumbers={false} showTag={false} language="python" className="max-w-2xl overflow-hidden m-auto" />
            <p className="text-center max-w-3xl m-auto py-5">
              Check out docs and examples
            </p>
          </div>
        </div> */}

      </div>
      <div className='bg-slate-50 px-5 border-t'>
        <div className='max-w-4xl m-auto pb-20'>
          <h1 className="text-4xl font-extrabold text-center pt-20 text-slate-900 mb-5">
            Frequently Asked Questions
          </h1>
          <FAQItem title={`Why not use Torch, TF, etc. directly?`}>
            Ideally, the ML framework used to run a model should just be an implementation detail. By decoupling your inference code from specific frameworks, you can easily keep up with the cutting-edge.
          </FAQItem>
          <FAQItem title={`How much overhead does Carton have?`}>
            Most of Carton is implemented in optimized async Rust code. Preliminary benchmarks with small inputs show an overhead of less than 100 microseconds (0.0001 seconds) per inference call.
            <br /><br />
            {`We're still optimizing things further with better use of Shared Memory. This should bring models with large inputs to similar levels of overhead.`}
          </FAQItem>
          <FAQItem title={`What platforms does Carton support?`}>
            <span>Currently, Carton supports the following platforms:</span>
            <ul>
              <li>x86_64 Linux and macOS</li>
              <li>aarch64 Linux (e.g. Linux on AWS Graviton)</li>
              <li>aarch64 macOS (e.g. M1 and M2 Apple Silicon chips)</li>
            </ul>
          </FAQItem>
          <FAQItem title={`What is "a carton"?`}>A carton is the output of the packing step. It is a zip file that contains your original model and some metadata. It <i>does not</i> modify the original model, avoiding error-prone conversion steps.</FAQItem>
          <FAQItem title={`Why use Carton instead of ONNX?`}>
            ONNX <i>converts</i> models while Carton <i>wraps</i> them. Carton uses the underlying framework (e.g. PyTorch) to actually execute a model under the hood. This is important because it makes it easy to use custom ops, TensorRT, etc without changes. For some sophisticated models, &quot;conversion&quot; steps (e.g. to ONNX) can be problematic and require validation. By removing these conversion steps, Carton enables faster experimentation, deployment, and iteration.
            <br/>
            <br/>
            With that said, we plan to support ONNX models within Carton. This lets you use ONNX if you choose and it enables some interesting use cases (like running models in-browser with WASM).
          </FAQItem>
        </div>
      </div>
    </div>
  )
}

const FAQItem = ({ title, children }: any) => {
  let [expanded, setExpanded] = useState(false);
  return (
    <div className='border-b flex flex-row'>
      <div className="text-xl font-thin mt-6 mr-3 text-sky-500 cursor-pointer relative" onClick={() => setExpanded(e => !e)}>
        <HiOutlinePlus className={`transition-all ${expanded ? "rotate-0 opacity-0" : "-rotate-90 opacity-100"}`} />
        <HiOutlineMinus className={`transition-all absolute top-0 left-0 ${expanded ? "rotate-0 opacity-100" : "-rotate-90 opacity-0"}`} />
      </div>
      <div>
        <h1 className="py-5 text-xl font-extrabold pb-5 cursor-pointer" onClick={() => setExpanded(e => !e)}>
          {title}
        </h1>
        <div className={`transition-all overflow-hidden ${expanded ? "pb-5" : "max-h-0"}`}>
          <div className={`transition-all prose max-w-none ${expanded ? "opacity-100" : "opacity-0"}`}>
            {children}
          </div>
        </div>
      </div>
    </div>
  )
}
