use clap::Parser;
use tokio::{net::UnixStream, io::{BufWriter, BufReader, AsyncReadExt, AsyncWriteExt}, sync::mpsc};

use crate::types::{RPCRequest, RPCResponse};

/// Connect to a UDS and return queues
async fn connect(path: &str) -> (mpsc::Receiver<RPCRequest>, mpsc::Sender<RPCResponse>) {
    let stream = UnixStream::connect(path).await.unwrap();

    // Split into read and write
    let (read_stream, write_stream) = stream.into_split();
    let mut bw = BufWriter::new(write_stream);

    let (incoming_tx, incoming_rx) = mpsc::channel(32);
    let (outgoing_tx, mut outgoing_rx) = mpsc::channel(32);

    // Spawn a task to handle reads
    tokio::spawn(async move {
        let mut br = BufReader::new(read_stream);

        loop {
            // Read the size and then read the data
            let size = br.read_u64().await.unwrap() as usize;
            let mut vec = Vec::with_capacity(size);
            vec.resize(size, 0);
            br.read_exact(&mut vec).await.unwrap();

            // TODO: offload this to a compute thread if it's too slow
            let req: RPCRequest = bincode::deserialize(&vec).unwrap();

            incoming_tx.send(req).await.unwrap();
        }
    });

    // Spawn a task to handle writes
    tokio::spawn(async move {
        loop {
            let item = match outgoing_rx.try_recv() {
                Ok(item) => item,
                Err(mpsc::error::TryRecvError::Empty) => {
                    // Nothing to recv
                    // Flush the writer
                    bw.flush().await.unwrap();

                    // Blocking wait for new things to send
                    outgoing_rx.recv().await.unwrap()
                },
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    // We're done
                    break;
                },
            };

            // Serialize and write size + data to the buffer
            // TODO: offload this to a compute thread if it's too slow
            let data = bincode::serialize(&item).unwrap();
            bw.write_u64(data.len() as _).await.unwrap();
            bw.write_all(&data).await.unwrap();
        }
    });

    (incoming_rx, outgoing_tx)    
}


#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    uds_path: String
}

/// Initialize the runner from command line args and return two queues to use to communicate
pub async fn init_runner()  -> (mpsc::Receiver<RPCRequest>, mpsc::Sender<RPCResponse>) {
    let args = Args::parse();

    connect(&args.uds_path).await
}