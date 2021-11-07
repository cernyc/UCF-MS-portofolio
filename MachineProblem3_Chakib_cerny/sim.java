import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

public class sim {

	public static int cycle;

	public static String operation;
	public static String type;
	public static String destRegister;
	public static String orgRegister1;
	public static String orgRegister2;

	public static int numoperation = 0;


	static ArrayList<String[]> instructions = new ArrayList<String[]>();
	static ArrayList<String[]> inProgress = new ArrayList<String[]>();
	static ArrayList<String[]> completed = new ArrayList<String[]>();
	static ArrayList<String[]> dispatch = new ArrayList<String[]>();
	static ArrayList<String[]> issue = new ArrayList<String[]>();
	static ArrayList<String[]> evaluate = new ArrayList<String[]>();

	static HashMap<String, Integer> states = new HashMap<String, Integer>();

	static HashMap<String, int[]> regU = new HashMap<String, int[]>();

	static ArrayList<String[]> reorder = new ArrayList<String[]>();

	static int IF = 0;
	static int ID = 0;
	static int IS = 0;
	static int EX = 0;
	static int WB = 0;

	public static void main(String[] args) {

		int sc = Integer.parseInt(args[0]); 
		int ways = Integer.parseInt(args[1]); 
		String traceFile = args[2]; String file = traceFile;
		

		// ways = 8;
		//int sc = 2;

		//String file = "src/val_trace_gcc.txt";
		Scanner scanner;
		try {
			scanner = new Scanner(new File(file));
			scanner.useDelimiter(" ");
			int x = 0;
			while (scanner.hasNextLine()) {
				String s[] = scanner.nextLine().split(" ");

				operation = s[0];
				type = s[1];
				destRegister = s[2];
				orgRegister1 = s[3];
				orgRegister2 = s[4];

				String[] inst = new String[12];
				inst[0] = operation;
				inst[1] = type;
				inst[2] = destRegister;
				inst[3] = orgRegister1;
				inst[4] = orgRegister2;

				inst[11] = Integer.toString(numoperation);

				instructions.add(inst);
				reorder.add(inst);
				numoperation++;
			}
			scanner.close();
			while (completed.size() < numoperation) {
				//System.out.println(cycle);

				x++;
				// FINISHED
				for (int i = 0; i < evaluate.size(); i++) {
					int latency = 1;
					if (Integer.parseInt(evaluate.get(i)[1]) == 1) {latency = 2;}
					if (Integer.parseInt(evaluate.get(i)[1]) == 2) {latency = 5;}

					if (Integer.parseInt(evaluate.get(i)[8]) + latency <= cycle) {
						String[] temp = evaluate.get(i);
						if (evaluate.get(i)[9] == null) {
							temp[9] = Integer.toString(cycle);
						}

						completed.add(temp);
						evaluate.remove(i);
						i--;
					}
				}
				
				// TO EX

				for (int i = 0; i < issue.size(); i++) {
					//if (cycle == 47 && i==1) {
					//	int breakpoint=0 ;
					//}

					boolean skip = false;
					int lat = 1;
					if (Integer.parseInt(issue.get(i)[1]) == 1) {lat = 2;}
					if (Integer.parseInt(issue.get(i)[1]) == 2) {lat = 5;}

					if (evaluate.size() <= ways + 1) {
							if (i > 0) {
								for (int p = 0; p < i; p++) {
									if ((issue.get(i)[3].equals(issue.get(p)[2]) && !issue.get(p)[2].equals("-1"))
											|| (issue.get(i)[4].equals(issue.get(p)[2])) && !issue.get(p)[2].equals("-1")) {
										skip = true;
										break;
									}
								}
							}
							if (evaluate.size() > 0) {
								String[] newest = issue.get(i);
								boolean hasDep = false;
								for (int e=evaluate.size()-1; e>=0; e--) {
									if ((issue.get(i)[3].equals(evaluate.get(e)[2]) || issue.get(i)[4].equals(evaluate.get(e)[2])) && !evaluate.get(e)[2].equals("-1") ){
										if (Integer.parseInt(evaluate.get(e)[11]) < Integer.parseInt(issue.get(i)[11]) 
												&& Integer.parseInt(newest[11]) == Integer.parseInt(issue.get(i)[11])) {
											newest = evaluate.get(e);
											hasDep = true;
										}
										if (Integer.parseInt(newest[11]) != Integer.parseInt(issue.get(i)[11]) 
												&& Integer.parseInt(evaluate.get(e)[11]) > Integer.parseInt(newest[11])
												&& Integer.parseInt(evaluate.get(e)[11]) < Integer.parseInt(issue.get(i)[11])) {
											newest = evaluate.get(e);
										}
									}
								}
								for (int c=completed.size()-1; c>=0; c--) {
									if (completed.get(c)[2].equals(newest[2])
											&& Integer.parseInt(completed.get(c)[11]) > Integer.parseInt(newest[11])
											&& Integer.parseInt(completed.get(c)[11]) < Integer.parseInt(issue.get(i)[11])) {
										hasDep = false;
										break;
									}
								}
								
								if (hasDep) {
									int latE = 1;
									if (Integer.parseInt(newest[1]) == 1) {latE = 2;}
									if (Integer.parseInt(newest[1]) == 2) {latE = 5;}
									if(Integer.parseInt(newest[8])+latE+1 > cycle) {
										skip = true;
									}
								}
							}
							
							if (!skip) {
								String[] temp = issue.get(i);
								if (issue.get(i)[8] == null) {
									temp[8] = Integer.toString(cycle);
								}
								evaluate.add(temp);
								issue.remove(i);
								i--;
							}
					}
					// else break;
				}
				// TO IS
				for (int i = 0; i < dispatch.size(); i++) {
					if (dispatch.get(i)[6] != null && issue.size() < sc) {
						String[] temp = dispatch.get(i);
						if (dispatch.get(i)[7] == null) {
							temp[7] = Integer.toString(cycle);
						}
						issue.add(temp);
						dispatch.remove(i);
						i--;
					}

				}
				// TO ID
				for (int i = 0; i < dispatch.size(); i++) {
					if (Integer.parseInt(dispatch.get(i)[5]) == cycle - 1 && dispatch.get(i)[6] == null) {
						String[] temp = dispatch.get(i);
						temp[6] = Integer.toString(cycle);
						IF--;
						ID++;
					}
				}
				// TO IF
				while (dispatch.size() < ways * 2 && instructions.size() > 0 && IF < ways) {
					String[] temp = instructions.get(0);
					temp[5] = Integer.toString(cycle);
					dispatch.add(temp);
					instructions.remove(0);
					IF++;
				}
				cycle++;
			}
			ArrayList<String[]> copy = new ArrayList<String[]>();
			copy = completed;

			for (int i = 0; i < reorder.size(); i++) {
				for (int y = 0; y < copy.size(); y++) {
					if (reorder.get(i)[0].equals(copy.get(y)[0])) {
						String[] temp = copy.get(y);
						reorder.set(i, temp);
						copy.remove(y);
						break;
					}
				}
			}
			
			for (int i = 0; i < reorder.size(); i++) {
				System.out.print(i + " fu{" + reorder.get(i)[1] + "} src{" + reorder.get(i)[3] + ","
						+ reorder.get(i)[4] + "} dst{" + reorder.get(i)[2] + "} IF{" + reorder.get(i)[5] + ","
						+ (Integer.parseInt(reorder.get(i)[6]) - Integer.parseInt(reorder.get(i)[5])) + "} ID{"
						+ reorder.get(i)[6] + ","
						+ (Integer.parseInt(reorder.get(i)[7]) - Integer.parseInt(reorder.get(i)[6])) + "} IS{"
						+ reorder.get(i)[7] + ","
						+ (Integer.parseInt(reorder.get(i)[8]) - Integer.parseInt(reorder.get(i)[7])) + "} EX{"
						+ reorder.get(i)[8] + ","
						+ (Integer.parseInt(reorder.get(i)[9]) - Integer.parseInt(reorder.get(i)[8])) + "} WB{"
						+ reorder.get(i)[9] + ",1}\n");
			}

			System.out.println("number of instructions = " + numoperation);
			System.out.println("number of cycles       = " + cycle);
			System.out.println("IPC                    = " + ((float)numoperation/(float)cycle));

			String name = "pipe_"+sc+"_"+ways+"_.txt";
			if (traceFile.equals("val_trace_gcc")){
				 name = "pipe_"+sc+"_"+ways+"_val_trace_gcc.txt";
			}
			else if(traceFile.equals("val_trace_perl")) {
				 name = "pipe_"+sc+"_"+ways+"_val_trace_perl.txt";
			}


			try {
				FileWriter myWriter = new FileWriter(name);
				for (int i = 0; i < reorder.size(); i++) {
					myWriter.write(i + " fu{" + reorder.get(i)[1] + "} src{" + reorder.get(i)[3] + ","
							+ reorder.get(i)[4] + "} dst{" + reorder.get(i)[2] + "} IF{" + reorder.get(i)[5] + ","
							+ (Integer.parseInt(reorder.get(i)[6]) - Integer.parseInt(reorder.get(i)[5])) + "} ID{"
							+ reorder.get(i)[6] + ","
							+ (Integer.parseInt(reorder.get(i)[7]) - Integer.parseInt(reorder.get(i)[6])) + "} IS{"
							+ reorder.get(i)[7] + ","
							+ (Integer.parseInt(reorder.get(i)[8]) - Integer.parseInt(reorder.get(i)[7])) + "} EX{"
							+ reorder.get(i)[8] + ","
							+ (Integer.parseInt(reorder.get(i)[9]) - Integer.parseInt(reorder.get(i)[8])) + "} WB{"
							+ reorder.get(i)[9] + ",1}\n");
				}
				myWriter.write("number of instructions = " + numoperation +"\n");
				myWriter.write("number of cycles       = " + cycle+"\n");
				myWriter.write("IPC                    = " + ((float)numoperation/(float)cycle)+"\n");
				myWriter.close();
			} catch (IOException e) {
				System.out.println("An error occurred.");
				e.printStackTrace();
			}

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
}
