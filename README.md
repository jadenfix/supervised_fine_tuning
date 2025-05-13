\section{Supervised Fine-Tuning of Code LLMs on NVIDIA T4}

\subsection*{Project Overview}
I'm training efficient coding-specific Large Language Models (LLMs) using Meta's CodeLlama-7b-Instruct as the base model, optimized for local execution on consumer-grade hardware. The implementation focuses on memory efficiency to run on \textbf{NVIDIA T4 Tesla GPUs} (16GB VRAM) using cutting-edge optimization techniques.

\subsection*{Key Components}
\begin{itemize}
    \item \textbf{Base Model}: \texttt{CodeLlama-7b-Instruct} from Meta
    \item \textbf{Hardware Target}: NVIDIA T4 GPU (Google Colab compatible)
    \item \textbf{Dataset}: 2000 samples from \texttt{codeparrot/github-code}
    \item \textbf{Training Framework}: \texttt{trl}, \texttt{peft}, and \texttt{bitsandbytes}
\end{itemize}

\subsection*{Memory Optimization Strategies}
\vspace{-0.5em}
\begin{minipage}[t]{0.48\textwidth}
    \begin{itemize}
        \item \textbf{8-bit Quantization} \\ 
        \texttt{BitsAndBytesConfig(load\_in\_8bit=True)} \\ 
        Reduces memory footprint by 4x
        
        \item \textbf{LoRA (Low-Rank Adaptation)} \\
        \texttt{r=8}, \texttt{target\_modules=["q\_proj", "v\_proj"]} \\
        97\% fewer trainable parameters
    \end{itemize}
\end{minipage}
\hfill
\begin{minipage}[t]{0.48\textwidth}
    \begin{itemize}
        \item \textbf{FP16 Mixed Precision} \\ 
        \texttt{fp16=True} \\ 
        50\% memory reduction + faster computation
        
        \item \textbf{Gradient Checkpointing} \\
        \texttt{gradient\_checkpointing=True} \\
        20\% memory saving at cost of 25\% speed
    \end{itemize}
\end{minipage}

\subsection*{Training Configuration}
\begin{lstlisting}[language=Python,basicstyle=\ttfamily\small]
# Memory-Efficient Training Arguments
TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # Effective batch size=4
    learning_rate=2e-4,
    max_steps=200,
    max_seq_length=256
)
\end{lstlisting}

\subsection*{Technical Stack}
\begin{tabular}{@{}ll@{}}
    \textbf{Quantization} & bitsandbytes \\
    \textbf{Efficient Training} & PEFT (LoRA), TRL \\
    \textbf{Acceleration} & CUDA 12.1, Torch 2.1+ \\
    \textbf{Monitoring} & NVIDIA-smi, Torch utils \\
\end{tabular}

\subsection*{Performance Considerations}
\begin{itemize}
    \item \textbf{VRAM Utilization}: Optimized to stay under 16GB T4 limit
    \item \textbf{Batch Strategy}: Gradient accumulation mimics larger batches
    \item \textbf{Warmup Period}: First 10 steps stabilize training
    \item \textbf{Checkpointing}: Full model save every 100 steps
\end{itemize}

\vspace{1em}
\noindent\fbox{%
    \parbox{\textwidth}{%
        \centering\textbf{Key Achievement}: Enables fine-tuning of 7B parameter models on single T4 GPU\\while maintaining original model capabilities
    }%
}
