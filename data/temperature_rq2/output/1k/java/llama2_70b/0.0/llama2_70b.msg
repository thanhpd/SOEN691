Added map view for error list
Replace 'private final Logger logger' with 'protected final Logger logger' in StatsdLoggerImpl.java
Logging configuration for transitive class dependencies updated from info to debug level.
Added utilities for tachyon.worker.block package
Update H2TestUtil.java to use a new random number generator for creating H2 database URLs
Added support for recognizing constraint violations on primary key with _id_ or .$_id_ in MongoDBDialect.
Added crawler initialization and navigation in CandidateElementExtractorTest.
Added logging for dropped frames in ReceiverUdpChannelTransport
Modified Util.fromStringList to use List.add() instead of List.set() for adding elements.
Discard private state in ChangelogStateHandleStreamImpl (FLINK-23139)
Update LSMSolver.java to adaptively calculate Rho and hardcode to GLM1 value for testing.
Update SecurityContextHolder strategy name to MODE_THREADLOCAL
Added constructor for JedisShardInfo with host, name, port, timeout, and weight parameters.
Fixed issue with loading initial data in StringStringSyncTest.
Removed unnecessary import statement for Symbol in NodeRepresentation.java
Added methods for loading and unloading data in MaterialDataPager
Removed @Test annotation from testDowngradeHoldCursorsUnderXa method in DB2TestServlet.java
Added support for transaction synchronization registry in JtaTransactionManager
Update exception type in testCommandOverridesDefaultIgnoreExceptions_nonIgnoreExceptionShouldBePropagated from HystrixRuntimeException to BadRequestException.
Added support for JSON format in OHttpResponse
Added decoration for search field and table in SearchDialogAction.
Removed unnecessary comment line in CapabilitiesDirectoryImpl.java
Added newline characters to the assert statement in CgIsaParamTest.java for better readability.
Replace assertThat with MatcherAssert in WaitStrategyTestUtil
Removed unnecessary comments and added Lombok annotations to App.java.
Removed BIAS_FEATURE from SentFeatureExtractor
Bump management version to 3.0.0

Change InetAddress.getLocalHost().getHostName() to InetAddress.getLocalHost().getCanonicalHostName() in TachyonClient.java
Removed unnecessary closePromise and replaced it with channel.closeFuture() in ClientConnectionsShutdown.java
Renamed class RefTest to extend PresentsTestBase instead of PresentsTest
Added support for replaying Aeron archives with query parameters.
Added verbose mode to FastClasspathScanner.
Removing unnecessary logging statement and replacing logger.info with logger.debug
Update to use mockery correctly
ContainerlessDockerDeployableContainer: updated to use CubeRegistry instead of hardcoded containerless Docker image
Fix TypeExtractor to correctly handle enclosed expressions
Added JillExtension.finish() and called it in Main.finish(), also added JillExtension.init() and called it in registerExtension()
Added import statement for List and updated array initialization
Fix for manual port exposure regression in 1.14.0 and enable Postgres participation in XA transactions (2PC)
Fixed typo in test suite name from 'GraohWalker' to 'GraphWalker' in XMLReportGenerator.java
Fixed date formatting in DurationBusinessCalendarTest
Modified Variant class to set empty alternate allele for missing values
Update VerifyBase.java to use a static variable for the user name instead of a hardcoded value.
Removed import of SpinnerAdapter in FeatureDemoActivity.java
Fixed sequence name quoting for PostgreSQL
Added synchronized block to avoid org.xml.sax.SAXException: FWK005 parse may not be called while parsing.
Added support for byte and short data types in ColumnValidationReader
Added UUID import and updated EndpointsLock constructor with unique identity name.

Remove unstable test case with Seraphim name in ConfirmTest.java
Replace getSystemResourceAsStream with getResourceAsStream in ApplicationClassLoader.
Added missing see reference for addSyncPacketListener and addAsyncPacketListener methods in XMPPConnection interface.
Fixed typo in Layer.java
Added option to skip count in MongoDBCollection.java
Replace \\ with \\\\ in source file path
Added started() method to Shell class
Added KeyboardBehaviorTests class to StyledTextAreaBehaviorTest
Modified JNDIManager.java to replace 'bind' with 'rebind' for sessionID and channel instance IDs.
Fixed issue with server tracer clearing current span
Update runFraction(float fraction) method in DeepLearningProstateTest.java to use a different fraction value (0.0007f instead of 0.001f).

AnimationFragment: updated target frame rate calculation from getFrameRate() to a hardcoded value of 60 fps.
Fixed encoding issue in AllureAttachAspects.java
Fixed check for OpenJDK tools jar existence
Added 'nobs' to excluded metrics in ModelMetrics
Added support for retrieving schema information for relationships
Update version number from 1.6 to 1.7.
Added setTick(0) to ControllableRunner to set tick to 0 when wantedTick is 0 and demandedTick is null.
Added try-catch block to handle exceptions when executing goal on deployment.
Added import statements for File and LifecycleException, and updated tomcat.setBaseDir() method to include the test class name in the directory path.
Fixed issue with TestIgnorance.java where services were not properly checked for ignorance.
Added test for string concatenation and array initialization
Modified ObjectUnderFileSystem.java to correctly handle listing of objects in a directory when the directory is not the root directory.
Added visit fragment spreads to VariableTypesMatchRule
Increased buffer size from 2 to 4 in SocketClient.java
Fix for DAO method getMaxInitialId() - changed return type from List<Integer> to Integer and removed unnecessary suppression of warnings
Fixed cache size calculation for image problems
Added import statements for Call and HttpMethod, and modified the RequestAuthenticator class to check the HTTP method before retrieving the parameter value from the request.
Disable Spark UI and update master to local for AnchoredPredictorTest
Added sleep statement before force stop for RealDevice
Update JavaFormatterTest.java to use new hash for formatted file.
Added @Column annotation to IClassWriter.myField
CodePrinter.java: Removed unnecessary space insertion when appending binary operator.
LEGGO MY EGGO. Update log message for when another node is master.
Added import statement for com.google.common.base.Strings and updated literals method to include the percentile value in the output string.
Added override for TestRule.apply() method in Expect class.
Remove unnecessary @Slf4j annotation from SiteAwareBuilder class.
Update English language rules: remove EnglishWordRepeatRule and add AvsAnRule, EnglishWordRepeatBeginningRule, and CompoundRule
Update Proto.java to use INT64 for boolean values and FLOAT64 for BigDecimal values.
Added GWT property for locale in CmsGwtActionElement.
Es6RewriteClass: Use source info from method for static member functions
Disable sound for upload service notifications
Added try-catch block for Exception in getChoice() method of AutoCompleteComboBox class.
Added protocol version number to KernelInfoHandler
Added method endTag() to class XmlEmit for ending a tag, and added method endEmptyTag() for ending an empty tag.
Added static method isSteamRunning() and isSteamRunningNative() to check if Steam is running, and updated init() and shutdown() methods to use these new methods. Also added a flag variable isRunning to track whether Steam is running or not.
Fix for exceeding 128 connections in Renjin128ConnTest
Added 'true' argument to getProcessInstance method to enable signal detection
ClientBuilder: Changed log level from debug to warn when multiple kubeconfig files are found.
Remove testSMILES method and update expectFormat method to use PDBFormat instead of SMILESFormat for data/coffeine.pdb
Added support for Conscrypt on Android platforms.

Fixed typos in AbstractPatternRule.java
Update import statement for FileDescriptor from core store API to common plugin API
Removed unnecessary RapidoidInitializer.initialize() call in Conf.
Remove unused import statement for 'MessageContent' and replace 'InputStreamEntity' with 'ByteArrayEntity' in MocoPostRequestAction.
Added import statement for AttributeExpression and added a new method attrX(Expression, Expression) that creates an AttributeExpression.
Update assertions for 'describe --tags' command to use 'lightweight-tag-1-gb6a73ed' instead of 'newest-tag'.
Remove unnecessary override of getScheme() in AbstractTFS.
Update Crypt.java to use a variable-sized buffer for decryption
Added test for different class loaders and closed specific cache manager
Added null check for shutdownNow() in ExecutorShutdownUtil
Added import statement for URISyntaxException and updated getFile method to use URI instead of toExternalForm
Modified ClientInterface to use core.aggregatorFragment and core.collectorFragment instead of aggregatorFragment and collectorFragment in AdHocPlannedStatement.
Update log4j configuration to change root logger level from DEBUG to INFO
Added connection ID to MpScheduler constructor
Removed unnecessary parameter 'message' from constructor of Violation class
Added tests for insert select null key reference and transaction
Added caching and updating parser for user agent strings.
Added tax ID and EORI number to exporter identification in CustomsDeclarationTest.
Update AccessLevel enum with new MINIMAL_ACCESS level and remove deprecated MASTER level
Added import statement for com.google.inject.ImplementedBy and added implementation of IIndentationInformation interface.
Disable anonymous access and CSRF protection for resource server
ScannerImpl.java: Fix for issue #12345 - Scan result file path is now correctly set when using a non-null path and non-null file descriptor.
Update glue path for PostgresDB tests
Added assertions for directory creation in TestDataContextUtils.
Removed unnecessary System.out.println statement in DataSetTableIterator.
Added test classes for multiplexing and demultiplexing

Added check for Docker machine installation before returning container IP address.
Fix connectivity creation with non-null context and manager
Added check for connectUri() != null in ClusterContext.accept(HostDetails hd) method.
Added initialization of clientName variable in RedisAppender to avoid logging during initialization
Revised ValidateFrame to use BorderLayout for layout management.
Added retry logic to RemoteRequestRetry class
Update JavaConstantDynamicTest to use assertThat with correct type
Fix grammar and wording in SubclassByteBuddyMockMaker.java.
Removed unnecessary imports and annotations, simplified ProcessInstanceIdCommand interface
Fixed issue with incorrect increment of invalid packets in RetransmitHandler.
Fixed illegal column index exception in Matrix.insertColumn and Matrix.removeColumn methods.
SONARPLUGINS-3097: Git blame improvement
Added import statement for java.io.Serializable and implemented Serializable interface in SimpleReadWriteLock class
Added missing putIfAbsent method call for Binding and Tracking in LifecycleManagerStoreImpl
Added support for multiple rules in CheckConformance by using a loop to iterate over a list of rules instead of hardcoding individual rules.
AllowableValues returned as LIST instead of string
Added checkSerializable method to Model class and added serialization test
Added listeners for join and leave events in DistributedGroup
Added parameter 'true' to SshClient constructor for password authentication.

Update copyright year in Encryption.java from 2021 to 2022.
Replace closeBatch with closePreparedStatement in Base class.
Renamed method 'browseFrow' to 'browseFrom' and added a deprecation warning for the old method.
Added support for materialized views in SegmentMetadataQueryRunnerFactory.

Updated confirmation dialog message for non-local database migrations
Modified DGLM class to update the calculation of p value based on enum values.
Added @Ignore annotation to testAnonymousType method in ClassReloadingStrategyTest due to failure on integration server.
Fixed JUnit4 seed handling to respect user properties.
Added exception handling to ResultExporter to reset image name on band element when generation fails.
ThriftCollection: Changed switch statement to use lowercase string comparison
Removed unnecessary @NotNull annotation from NotificationsManager.getEmailPath method.
Added method assertFloatsEqualDefaultTol to JglmTesting class
Added leading slash to non-empty path annotations
Added support for time gauges in JHipsterMetricsEndpoint
Added DELTA constant and updated quality check in ResizeSettings
Added check for memoryCache being null before scheduling updates in MainActivity.
Update Message.java, version 1.117 -> 1.116

Updated stop timeout unit from milliseconds to seconds in AlluxioMasterProcess.
Increased the size of the motion search region from 200x3 to 220x3.
Added validation for large requests in ResponseParser.
Added Windows drive letter handling in classpath string array splitting

Added STATUS_TOO_MANY_REQUESTS constant and updated response status code in outputThrottleError method.
Update logging level for copying addon files from INFO to FINE.
Added method clearViewCached and updated lookForChildViewWithId to use sparse array
Increased test duration from 20 seconds to 30 seconds in ArchiveRecordingLoadTest.
Added Flag.CACHE_MODE_LOCAL to DistributionManagerImpl
Added methods for boxing and unboxing integer arrays
Added import statement for android.util.Log and modified exception message to use Log.getStackTraceString(throwable) instead of throwable.toString()
Added new rule for English language: PRP_RB_NO_VB
OperaDriver.java
Update RadarLevel2Servlet.java to change dataType from Station to Radial and update dataFormat to NEXRAD2
Added call() method to HamcrestCondition<T> matcher and isTrueOrNonNull(Callable<T> item) methods.
Removed unnecessary functionLibrary.load() call in OMetadata constructor.
Added type variable cycle test with TB annotation.
Fixed issue with cache size configuration in Jinjava
Fixed nan handling in Sum.java
Added cluster configuration entry permissions.
Fixed null pointer exception in ApnsConnection constructor by adding a check for null listener.
Replace HashMap with ConcurrentHashMap in Finalizer
Update GaussianProposalDistribution scalar value from 20.0 to 0.1 in LinearRegressionTest.java
.proxy.ProxyHandler.java
Updated BaseSource.java to use ConfigurationKeys.DEFAULT_WATERMARK_VALUE as default watermark when there are no previous committed states found.
Modified TimerTask.java to calculate next timeout date correctly.
Added support for Java 7 and 8 features.
Added tearDown method to X509ProcessingFilterTests
Fixed bug in ExpressionAnalyzer.java: removed SuspiciousMethodCalls warning and added proper null checks for scope variable.
Update Validator.java to use wildcard type parameters in CheckRange class.
Increased number of approvals required for AICS process from 2 to 4.
Added import statement for reactor.netty.internal.util.Metrics
Added rethinkDB create database functionality
Added ClassLoader manipulation for proper web.xml parsing
Modified GenericMultipleBarcodeReader to add support for offsetting result points.
Update JapaneseElementTest to use 'R' instead of 'H' for two-digit year format.
Added Executor to Async class and updated run method to use it.
Added logging for PolicyFinderModule serious failures and optimized policy parsing for performance.
Update TraversalStrategyPerformanceTest to use 75% assertion percentile instead of 95%.
Fix for terminology service: Remove unnecessary check for total count
Fix issue 122: handle end date correctly in PeriodCounterFactory
Reverted last available chunk id from %s to %s and added check for new chunks
Added tests for pool claiming with TTL and allocator validation
Added getForm and setForm methods to AbstractFormPropertyRenderer to access the Vaadin form that contains the renderer.
Removed unnecessary dependency on icu4j in SdkConfig.
Update connectivity manager to use application context instead of system service
RafMetadataReader: skipping 2 bytes in IFD pointer parsing
SpanBuilder: Start a new span and set it as the current span.
Added transient field 'photo' to document class Profile and updated test to reload from cache and disk.
Update XmlExtractorHelper to use 'of' instead of 'absent' for Optional<InputSource> return type.
Added constructor and setter methods for className and service in ProxyFactoryConfig.
Added setResultIdx(1) to Row nodeRow = getRow(new Cell(1, node))
Removed unused import of SimpleDateFormat and updated snapshot and transaction serializer.
Fix for Arrays.asList serialization issue (JRE bug 6835580)
Added check for 'value' flag before retrieving input field value
Added TLS hostname verification for XMPPTCPConnection.

Increased wait time for InteractiveTestJob from 5 seconds to 10 seconds.
Added linear algebraic matrix multiplication method to VMath class.
Fixed calculation of termId in ArchiveConductor
Fixed VerifyMojo.java to correctly check for orphans
Added check for existing project directory before creating a new one in EditorUtilities.copy method.

Added import statement for 'io.aeron.exceptions.AeronException' and updated exception handling in stopReplication method.
Added styling to day view for odd resources
Modified CmsDateBox.java to set popup modal flag to true
Fixed issue with many references in BatchLinkingService
Updated SimpleRoundRobinLBTest to use a shorter ping interval and added a call to ensure that the ping cycle has kicked in and all servers are set to alive before printing the server list.
Added ability to handle extra artifacts in SingularityMesosTaskBuilder
Update RunLocalPayloadTest to use a temp directory for local payload tests
Removed unused import statement for Logging
Added pageStoryId and fullSizeSource fields to Photo class.
Added support for Linux-specific native library loading in AttachContainerCmdExecTest.
Added methods for waiting and timeout handling to TestCommons class.
Added functionality to update the current selected profile in the selection list when switching profiles in the AccountHeader.
Added query string support to RackFilter's redirect functionality.
Added null checks for registeredService and service in DefaultMultifactorAuthenticationTrustedDeviceBypassEvaluator#shouldBypassTrustedDevice
Added space between 'FailedExecutingTask' and runnable in TimeScheduler3.java
Replace ConnectionAcquireTimeoutEventListener with ConnectionLeaseTimeThresholdExceededEventListener in FlexyPoolConfiguration.
Added import statements for Collections and HashMap, and modified getDeclarations method to return an empty map if declarations array is empty.
Fixed detection of exceptions that can't possibly be matched.
Added warning for possible deadlock scenario in ZipExportDelegate
Fixed issue with program invocation in ClusterClient.java
Remove @Incubating annotation from StackdriverMeterRegistry class
Updated Contents.java to use ROOT_PATH instead of hardcoded '/' for selecting the root folder in the repository.
Added request content retrieval method and stored it in a variable named 'body' for future use.
Added method resolvePath to ResourceResolver to handle relative resource paths.

Removed unused instruction WIDE and updated operand encoding for IIINC instruction.
Added health check requirements for Elastic Load Balancer
Fixed RedisModuleMemberRegistrationWithTTLConfiguredIT test to use RedisDialect.toBytes instead of getDialect(provider).toBytes
Increased length of DefaultChannelId's newLongValue() buffer from 4 to 5 for better performance.
Added return statement for getOwners and getOwnersExplicit methods in TypeService class.
Added import statement for InetAddress and updated host variable to use getLocalHost().getHostAddress() method.
Update WordTokenizer to use getTokenizingCharacters() instead of hardcoded TOKENIZING_CHARACTERS
Added support for gzip file conversion and updated policy providers.
LZ4BlockInputStream.java: Fix for compression method LZ4, compressed buffer length check
Added methods to get task and block speeds in Listener4SpeedAssistExtend.
Fixed bug in LiteBlockingWaitStrategy.java by removing unnecessary lock and condition wait
Added fullRedrawHint variable to DefaultScreen.java and set it to true.
Fixed bug in NioWorker.java: handle AsynchronousCloseException properly
FIXME Why is this failing to proxy the type
Added sorting to CollectorContextTest
Added whitespace handling in PRTokeniser
Added method isInputMethodTarget() to assert that a text view is an input method target.
Fixed injection of DropwizardModule and InjectorWrapper
Removed transient modifier from Callback field in RangeState

Fixed hack for delegated tasks
Modified Parser.java to use a limit of 2 when splitting strings with ':'.
Added support for locale-specific AreaFinder names
Added stream wait timeout to StreamInterceptingTunnel
Fixed format of elapsed time in ThreadLog to use nanoseconds instead of seconds
Added private constructor to Cycles class
Replace ClusterExecutor creation with SecurityActions.getClusterExecutor(cacheManager) for better security.
Added @Ignore annotation to EdgeTest.java to ignore test cases in Edge browser.
Temporary hot fix for SPARQL_LANG equality issue
Added support for acyclic joined graphs in the collate API endpoint
Added import statements for HashSet, UUID, and PersistentObject, and updated PushApplication class to extend PersistentObject and implement PushApplication interface.
Added methods to set separator and quote for CSVExporter, and updated export method to use them.
Update FlowableReverse.java to use flatMapIterable instead of flatMap for reversing lists.
Removed unused method registration for subsystem deployment writer in ApplicationClientConfigurationPersister.
Added 'expiry' field to Cookie class and added methods to get and set expiration date.
Add support for reading UNION data types in ORC files
Added suppressFBWarnings annotation to ofBootLayer method in ClassFileLocator.java
DeployMojo: Set createExternalUrls to false by default and add exposecontroller as a dependency
Added initialization of 'initialized' field to IOSServer class.
Modified HitbtcTradeServiceRaw.java to use order.getId() instead of order.getType() + createSymbol(order.getCurrencyPair()) + nonce for creating client order id
Added method getTransactionController() to class PersistentPriorityProcessorPool.
Added WebSocketsExample with low-level WebSocket functionality.

Added support for HTMLUnit in DesiredCapabilities
Replace MDC.put(LoggingListener.JOBNAME, ... with MDC.put(LoggingListener.JOBLOG_FILENAME, ... and MDC.put(LoggingListener.JOBNAME, jobExecution.getJobConfigurationName());
Replace 'model' with 'this' in JadeTemplateEngine#get
]}  Add check to disable waiting for ajax attributes in RichFacesTest.
Update DriverConductor.java to use CommonContext.IPC_CHANNEL instead of subscription.uri() for receive channel endpoint creation.
Added blank check for TextComponent
Removed unnecessary import statement for 'javax.annotation.Resource' in BaseXMLBuilderTests.java.
Added method registerEventListener(EventListener listener) and registerFilter(filterClass, urlPattern) to ApplicationContext class.
Renamed 'getDownloadableId()' to 'getId()' in DownloadFromUrlInstaller and updated references.
Fixing bug in SnappyInputStream.java: handling incorrect chunk size
Added support for loading font icons from files in addition to assets.
Added ability to add isotopes with charges to MolecularFormula.
Added StaxEvent array initialization
Update AtomicFileWriter.java
Added fast QR Code detection configuration
Added check to only locate codebases required for analysis if there are discovered codebases
Fixed trimTimeAnno method to handle undefined time values and added a new method to split time into arrays.
Renamed ProcessEngineService to PlatformService in FoxPlatformDemoDataDeployer
Added user agent checking for mobile theme override
Added BankInterruptedException when robot dies of old age
Added support for Samsung Electronic's orientation sensor type (65558)
TriggerTask: Changed workQueue.add(logEntry) to workQueue.put(logEntry) for better performance.
Fixed incorrect type casting in ShellErrorHandler.java
Configure tsd.query.allow_simultaneous_duplicates to true
Removed unused 'keepAliveTimeout' field and updated 'allowCoreTimeout' checkbox
Added 'endpoint' field to ClusterEventDissector.
Added error handling for getting input type information
Added flyWeightTasks queue and onEnterFlyWeight/onLeaveFlyWeight methods to QueueListener.
Added debug check for resolve status in SimpleConfigList constructor
Added slow test annotation to IsStillRunningServiceTest.
Update PresentsClient.java to version 1.69, closing stale connections.
Added support for setting the runtime root folder in LocalTestEnvironment
Added constant initializer for 'hashCode' in UnsafeBuffer
Fixed issue #12345: Implemented CONDITIONAL_EXPECTATION for RandomVariableDifferentiableAAD.
Added error handling for wrong content-type in AbstractSingleImageLayer
Disable printing of hedge error distribution and reduce number of paths for BlackScholesDeltaHedgedPortfolioTest.
Update seat proxy binding version from V4 to V3
Added test for handling command line arguments in Main class.
Update Shard enum with new Garena path
Added method getReceivedMessages() to AbstractReceptionBean.java
Added logging statement in loadMetadata method of BaseFileSystem class
Added logging and handling for ClassNotFoundException and LinkageError in ServletContainerInitializerRegistrationBean
Replace TupleClassMeta<?> with TupleClassMeta for better type inference
Version bump to 3.1
Added reflexivity checking for indexed property chains.
Added position() method call before writing to file channel in AsyncFile class.
Added check for duplicate Ands in LogicTransformer
Added support for versioning to GrailsDomainClassPersistentEntity
Added check for availableSequence >= nextSequence in BatchEventProcessor to prevent unnecessary calls to onBatchStart
Added test for signers having CA bit set
Added isMainMethod() to check if a method is the main method and skipped analysis of main methods in FindOpenStream.
Increased queue size of AsyncUserAccessAuditLogWriter from 100 to 10000
Modified equals method in Main class to also consider external flag
Update testNonExistentHost to use invalid hostname instead of non-existent host
Added null check for error stream in SegmentHTTPApi

Fixed logging of dropped errors in Operators.java
Added field 'fUniqueId' to class Description and updated constructor to include unique id parameter.
Update HALBuilderTest to use 'get' instead of bare attribute match
Added support for multiple place types in NearbySearchRequest
Added code to close selector and server when WebSocketServer is interrupted
Added check for even number of fields in ODocument.fields(String, Object, Object[]) method
Modified AccessSetUpdate.java to use executeWithoutAccessCheck() instead of execute() for deleting current links to access types.
Fixed issue with selecting options in BrowserTest
Fixed getServerVersionNum() method to correctly return server version number.
Added gossip initialization check to NodeCmd.printInfo() method.
Renamed shader file path to exclude 'shader/' prefix in ShaderSelectFilePropertyHandler.
EstimoteBeacons.java: updated BeaconConnection constructor to use a different parameter for connecting to beacons
Fixed header height calculation to account for all-day events
Added override of toString() method in PostmarkResponse class
Fixed bug in ServerModel.java: Allow re-association of HTTP contexts only when allowed
Added check for atom name and element in HetatomImpl
Added support for SSL/TLS connections by catching SQLException and IOException.
Fixed issue with incorrect data type in LivecoinAdapters.java
Added logging of profiler configuration in OProfiler#acquireExclusiveLock() [OProfiler.java~113-114]
Fix for creating null blocks in MapFlatSelectiveStreamReader
Remove doapply and apply methods from DBCollection
Added import statement for Sort in Batch2CoordinatorIT.java
Fixed exception handling in JerseyServerSentEvents.java
Reverted AddItem method to reAddItem and added a warning about adding items not previously on the list.
Added checkClosed() method call in LockFreeCache.java to ensure that the cache is not closed during lookup
Added a new completion candidate for address operation separators
Allow PeerGroup to manage download peer statuses
FollowerState: removed unnecessary last voted for candidate reset and introduced semi-random election timeout.
Update EC2 version from 2012-12-01 to 2013-02-01
Fix for GitRepository.java: Remove unnecessary null check for 'executor' field.
Update RollbackTest.java to add author name
Added @DatabaseField(generatedId = true) annotation to AllTypes.id field.
Catching Throwable instead of Exception in KafkaSupervisor's handle method
Added ignore regexes for Libnd4j assign error and CIFAR-10 GAN issue
TargetType.java: Clarified documentation for METHOD_RETURN and added missing METHOD_PARAMETER annotation.

Update JDBC driver version to 1.3.5
Replace Throwables.propagate with RuntimeException
Added retry mechanism for APNS push notifications
Added methods to ResponseWrapper class to match the Response interface

Update UmlautTest.java to use Java 7's Enumeration instead of JDK 1.4's Enumeration<? extends ZipEntry>.
Added support for PUT requests by overriding doPut() method and creating a new HttpPutRequestHandler class.

Modified URLPackageScanner.java to correct class name extraction.
Added synchronization using CountDownLatch for job execution in JobManagerTest.
AzureOSImage: Fix for incorrect image name when version is present
Added two new lines of code to the WireHandler interface to handle exceptions.
SKIPPED printing for T_ANY or subtypes
Disable use of new byte code generator
Update ALLUXIO_DOCS_URL to use major and minor version numbers.
Added support for passing additional closeables to Lifecycle.trying() method.
Modified JPerfmonParamsDialog.java to update the way metric radios are initialized.
Fixed documentation for PdfBoxGraphics2D.createCopy() method.
Added null check for attachments map in UnsavedRevision.addAttachment method.
Update font size for language label in SpellAndGrammarCheckDialog
Added tests for PathService.parsePath() and Path.toString()
Fix for Aesh bug: Removed unnecessary check for variable name ending with '=' in ShellImpl.
Added GET method to SavedRequest in AbstractProcessingFilterTests
Fixed bug in ImmatureClass.java by adding missing equals and hashCode methods.
Emitter thread got interrupted, shutting down.
Added import statement for rx.Observable and implemented hasObservers and observable methods in EmptyInventory class.
Remove @SuppressWarnings annotation from getScrollListWindowHeight method in ViewFetcher class
Added support for tracking currently evaluated file with a Deque instead of a Stack
Disable default configuration for WebSecurityConfigurerAdapter
Removed unused import and method, and simplified ClusterTests class.
Added method createTempFile() to TestCommons class.
Modified EmployeeDAO class to add new insert and upsert methods, and updated delete method to accept additional parameters.
Update toPatternString and toString methods in AbstractDSLMappingEntry to use getSymbol instead of hardcoded section name
Added throwing of InternalServerErrorException when unable to load roles
Increased maximum processing time for JobServiceJmsIntegrationTest from 2000 to 5000
Added 'onload' attribute transformation to FastArchivalUrlReplayParseEventHandler.
Reverse stoichiometries list in Subunits class.
Added programmatic signaling of shutdown signals and removed barrier from shutdown signals.
Remove JAVA_OPTS environment variable from ProcessBuilder
Added check for index being less than 0 in RoundRobinExecuteWithFailover
Added JsonCreator to GrokPattern for creating instances with name and pattern
Fixed error message in MarkupEngine.java to correctly indicate parsing is skipped when meta data is missing.
Added support for loopback addresses in ParsedInterfaceCriteria
Fix bug in NodeUtil.java: incorrect checkState message
Added interceptor priority to CustomPathResolver
Added configurable option for using internal network in TermdBuildDriver
Added suppression of warnings for unchecked cast in fromJSON method
Renamed EvaluationException to PositionedException in Exec.java
Added logging for missing Kafka topics

Allow configuration string to be empty in ClusteredObservationBus constructor
Added warning for starting PeerGroup with no attached block chain.
Added UFS attributes to WebInterfaceGeneralServlet
CorrectEventFiringTest: Changed \'driver.findElement(By.tagName("id-name1")).click();\' to \'driver.findElement(By.id("id-name1")).click();\'.
Removed unnecessary import statement and Logger reference in IndexedSet.java
Fix handling of error codes in Finder.handleErrorCode

Update Util.java to version 1.51: change in loading class from current thread's context class loader to current instance.
Added directory existence check and mkdirs() method call in DnsCacheFile constructor.
Removed unused variable 'postEffectShaderManager' in GVRViewManager.java
Added import statement for org.mozilla.javascript.ContextFactory in Scripting.java
Added mapping for COL_TYPE.DECIMAL to Types.DECIMAL
Removed unnecessary stopUFS method
Replace 'DefaultFullStompFrame' with 'DefaultStompFrame' in toString() method of DefaultStompFrame class.
Added a warning and skipped computation for getExpectedLogBelief method in ProjDepTreeFactor class.
Removed unnecessary 'e.printStackTrace();' statement in DiscoverArchiveManifestFilesRuleProvider.
Added marker title and snippet to PlacesSampleActivity.
Update synonyms batch request to include forwardToSlaves and replaceExistingSynonyms parameters
Added hashCode and equals methods to Resource class.
Replace List<Stats> with ImmutableList<Stats> and List<PairedStats> with ImmutableList<PairedStats> in StatsTesting.java
Added constant NOT_A_CHAR for invalid Unicode characters
Renamed TAG_H_POSITIONING_ERROR to TAG_HORIZONTAL_POSITIONING_ERROR in GpsDirectory.java
Deprecate ZMQ.poller() and add new constructor for creating a poller without a context.
Added test case for spaces in LDAP URL
Added test for cell timestamps in TestFilters
Fixed issue with GroupProperty.java, now correctly handling null values.
ClientTest: Changed consumeAfter to andThenConsume and removed atLastConsume
Update SpotSceneDirector.java to version 1.26: added logging for cluster changes
Removed unnecessary test method 'testTransactionOptimisticCacheMgmt2Db'.
Remove unused import 'org.apache.flink.configuration.description.Description' in MetricOptions.java
Update LOW_STORAGE_SPACE_THRESHOLD_DEFAULT from TERM_MIN_LENGTH to SEGMENT_FILE_LENGTH_DEFAULT

Update default servlet version from 2.4 to 2.5 in Metadata.java
Added try-catch block for quitting drivers in canStartMultipleIeDriverInstances method.
Modified GrailsDomainBinder.java to set optional property to true for collection role.
Changes to CmsGalleryField.java: removed unnecessary variable and updated gallery service instance.
Added support for Catalan language by adding ç, è, é, ì, í, ó,ò, ú,ü, ÿ, and à to the list of valid characters in WordListValidatorTest.
Added DEFAULT_GROUP_ID constant to FractionList
Updated SingularityS3UploaderConfiguration to use a minimum upload size of 5242880 bytes instead of 0 bytes
Added call to 'writeContext.flush()' in OffHeapValueHolder.writeBack()
Expand UPC-E for purposes of searching
Changed toString method in BoxedUnit class from () to {}.
Added wait statement for browser environment in JasmineTestRunner.java.
Added getters and setters for enabled and registerManagement properties in AutoServiceRegistrationProperties.
Update MessageSerializer to use Header.serializer().serialize(t.header, dos, version) instead of Header.serializer().serialize(t.header_, dos, version)
Added support for exploding geometry columns with non-zero SRID
Added logic to invalidate the root drawable only when the Lottie animation changes.
Added map from lock ID to lock record, and added initialization of lock array in constructor.
Added readCurrentBlockToEnd method to FileInStream class.
Remove unreferenced temporary builds
Added stop() method to FileCacheStore to properly shutdown executor
Added code to copy required properties for service chaining from incoming carbon message to response carbon message in TargetHandler.
Update build start date handling in ExtractorUtils
Remove unnecessary null check for ServletContextImagePicker
Added filter to ignore messages to non-existent vertices in SparkExecutor


Remove unused method checkActiveController and simplify chooseController
Added support for generating typed externs in DefaultPassConfig.
Added capacity check and cleared in-memory session records when flush records to file.
Added setTitleColor method and updated invalidateTextPaints to use mTitleColor

Added support for case-insensitive regular expressions in Element.java.

Added sleep statement in AbstractMindmapsEngineTest.
Added import of SecurityContext and updated PartitionCallableRequest to use secure callable when security context is available.
Added error handling for internal failures during authentication
Added methods ctorSuperS, ctorThisS, ctorSuperS(), and ctorThisS() to GeneralUtils class.
ORecordAttribute: Changed getter for @class from Element.flatMap(r -> r.getSchemaType()) to Element.toElement().getSchemaType()
Update TransactionConfidence.java to change confidence type from OVERRIDDEN_BY_DOUBLE_SPEND to DEAD when transaction becomes newly dead
Added logging and handler information to LoggingIntrospector
Renamed abstract class DefaultIndexRedirect to implementation class.
Added support for signing requests with OAuth tokens.
Removed unnecessary System.out.println statement in SomeSteps constructor
Added onStart() and onResume() methods to RubotoActivity.
Modified UserImpl to use accounts/{ACCOUNT_ID}/users endpoint for user creation
Update setUp method to setUpChrome and add assumption for ChromeDriver
Clear out speak service and remove shutdowner from queue
Modified SeleneseTestCase.java to change 'runtimeBrowserString' from a private method to a protected method and updated the signature of 'setUp' accordingly.
Removed unused import com.google.common.collect.Sets
Added support for using Maven pom properties to specify the ReleaseId for a KieProject
Added private modifier to lockPostingChangeNotifications field
Added comparison of task level in SingularityTaskMetadata compareTo method.
Added definitionId field to CaseInformation class and implemented getDefinitionId method.
Added import statement for org.apache.commons.lang.exception.ExceptionUtils and updated catch block to log and rethrow exceptions
Removed unused methods setSize, setOnColor, and setOffColor from ToggleSwitchBase
Update EsMetricsAccessorTest to use node.getHost() instead of node.getContainerIpAddress()
Added notifyOnPreferenceHeaderSelected() call in onPreferenceTreeClick() and onPreferenceHeaderClick()
Removed unnecessary 'throw' statements in Fmrc class.
Added a caveat for the CacheCreateConfigOperation when createAlsoOnOthers is true

Deprecate Closeables.closeQuietly() and add deprecation warning
Fixed bug in SessionListenerBridge: now properly resetting session in tearDown() method.
Added fast atan2 implementation for better performance.
Modified HDFSDataset.java to return a new Builder instance instead of creating a new dataset directly.
Added method 'shouldRun' to check if a persistent timer is already executed from a different instance or should be executed.
Replace StringBuffer with StringBuilder in convertToHex method
Modified IntervalTree.java to change the order of stream concatenation in the overlap method.
Refactor TestHierarchicalMethodsSubclass to use private method with boxed type parameter
Fix compiler bug by adding a work-around for closeable processor class loader.
Update DistributorFlags to use 'reject-unsupported-caps' instead of '--reject-unsupported-caps' as the config value name.
Added functional interfaces for Throwing
Modified BaseClassFinderService to use ClassFinderActivator.getPackageName() instead of resourcePath
Added SwingUtilities import and updated showProgress() method to use invokeLater() for better performance.
Added new methods for looping sounds with gain and position control.
Removed unused resource 'data' from interface Resources.
Added Apache License header to PlanConfiguration.java
Added double quotes around thread names in ThreadDumps for better readability and consistency with other parts of the codebase.
Added logging for Class.forName errors in ReflectionUtil
Replace CLEANUP_INTERVAL with CLEANUP_MASK and modify cleanUp() method.
Update supported source version to 18
Added bounds checking for LSMSolver.
Added retry mechanism for HTTP 401 unauthorized requests
Removed unused import statement for LrsObject
Renamed method 'settings' in interface 'NamedSubResourceSettingBuilder' from 'restSetting' to 'RestSetting' and removed unnecessary parameter 'final'.
Fixed issue with creating upload sub-directory under server temporary directory. Changed mkdir() to mkdirs() to create parent directories if they don't exist.
Added Jackson annotations to ErrorMessage class
Added grace period option to TemporaryJobBuilder
Update TestOperaDriver.java: remove unused fields, set default values for controlWindow and currentProduct
Set DEBUG to false
Added isSupervised method to DeepLearningModel class.
Modified NewSessionTest to check screenshot dimensions using assertTrue instead of assertEquals.
Added 'sudo' method to DBClosure class to simplify execution of database operations under admin user
Added volatility to secret field in CredentialReference
Added method isFakeBugType(String) to BugInstance and modified constructor to use it.
Added user-defined field support to ControlFile
Fixed bug in CmsSolrIndex.java - now properly handles container pages
Removed pax-exam-junit4 and pax-exam-spi dependencies from WarTestProbeBuilderImpl.java
Update DataHeaderFlyweight.termId(activeTermId).termOffset(termOffset) to dataHeader.termId(activeTermId).termOffset(termOffset)
Added constructors and default values to UnitizingStudy class.
Cast termLength to long to avoid integer overflow in LogBufferDescriptor#calculateFileSize
Added error message for null scheme in FileSystem constructor
Added @Ignore annotation to testCacheExpiration method due to concurrency issues.
Added RemoteWebDriver to WebElementConverter and updated createCachingWebElement method to set correct parent and file detector.
Added MissingResourceException handling to ConstantsWithLookupProxy
Added LinkedHashMap for property ordering in FlatProperties
JDBC_PING: Added debug logging for initialize_sql statement execution

Added import of AlwaysIncreasingSystem2 in CreateActionTest
Fixing Utils.java: updating write method to handle length correctly
Remove deprecated method isSubscibe() and replace with isSubscribe() in ConfigureForm.java
Added Utils and Records to reserved type names in JavaCodeGen.
Added trace logging for marshaled payload in SoapClient#marshall method.
Merge parts into output with different paths
Update SimpleParameterList.java to version 1.12, adding a check for at least one output parameter.
Fixed test FlowRunnerTest2 to use FlowRunnerTestUtil.startThread and added join method call to ensure runner thread is finished before asserting flow status.
Fixed incorrect comment syntax in JSpecRunner.java
Added override for fillInStackTrace() method in ObsoleteVersionException class to avoid stack trace overhead.
Config.java: Set VANILLA to false
Added completed limit orders to open orders list
Modified ToString.java to call superclass's toString implementation as part of the generated equals algorithm.
Rename file from 'newFile' to 'oldFile' in OfflineChangeLogHistoryService.java
Added ability to repeat last command by pressing up arrow key.
Update ServerEventTest to use InetAddress.getByAddress() instead of InetAddress.getByName() for resolving hostnames.
Removed unnecessary parent block from ParserUtils.java
Added import statement for 'InputStream' in CoverageTest.java
Renamed 'performing' to 'committing' and 'committing' to 'committed' in SessionSummary class.
Fixed issue with resampling precision loss by rounding result instead of floor() and updated exception message to reflect the change.
Update DefaultScaffoldRequestHandler to use domain.update() method
Modified GradleCli.java to check for GRADLE_COMMAND in environment variables
Added support for monitoring memory usage and updated sensor name for CPU usage
Added enabled status for injectable headers all requests checkbox
Upgrade map to ConcurrentHashMap for better performance.
Modified PathRetractionStrategy to correctly handle empty steps.
Renamed service text field from 'myService' to 'mySample' in CreateServicePage.java.
Removed deprecated methods getHttpServletRequest and getHttpServletResponse from Context interface
Ask for user confirmation in case of number of files to convert > 1000
Added code to resolve app name and handle errors when launching app from RhodesService
Fixed leader appointer startup block
Added method tryGetDefault() to SharedHealthCheckRegistries class
Fixed hashCode() implementation in JDateTime to use 'time' instead of 'jdate'.
Update Patterns for Nullable and NonNull Annotations
Added static create method to CollectionSubject
Renamed toStringEquals to strEquals and updated method signature to use Objects.equals instead of == for better performance and null-safety.
Update authentication mechanism name from 'PLAIN' to 'BASIC' in SecurityRealmService.
Added initialization of sipStack and user agent for test using stun server
Added capability proxy settings to NewSessionPayload
Added Google Guava's Throwables.propagateIfPossible() for better error handling in DefaultLogicResult.
Added test case for ProxyServlet with query parameter 'note:Leitbild' (Issue #4).
Added test for ArrayProperty.assertArrayType(null) being null-safe
Added import of ServerRole and updated rename test to work with single server mode.
Added ability to interrupt long-running queries with Ctrl+C.
Added isRuntime method to ThreadPoolResourceDefinition
Rename import from com.google.common.base.Stopwatch to com.google_voltpatches.common.base.Stopwatch
Added error handling for script elements with parse errors.
Modified AndroidTestFinder.java to correctly identify instrumentation test classes for InstrumentationTestRunner
Added ResourceAccessor field and getResourceAccessor() method to BaseLiquibaseTask.
Update ResourceConfigApplier interface to use MocoConfig instead of final MocoConfig
Update Loader.java to use StandardCharsets instead of Charset for reading files
Fix issue with Postconditions.java: remove unnecessary return statement and add throw statement for better error handling.
Added attribute list to BodyCells class.
Added support for watching template and asset folders in addition to content folder.
Set log level to debug in LoggingTest.
Added SysUtil.dumbDeleteRecursive() call to handle non-existent files in UnpackingSoSource.java
Update PublicationImage to use a more accurate method for calculating active transport count.
Fixed bug in MultiUserChat.java where userHasLeft() was not being called before leaving a room.
FastAdapter: Changed logic for calculating total added items
Update PinListSync to log warning instead of error and add debug logging.
Modified S3_PING.java to use HTTPS instead of HTTP for bucket requests.
Update copyright year in RawType.java from 2010 to 2011
Fixed implementation of CloseableAdapter.onClose() to return Mono<Void> instead of void.
Replace 'java.util.concurrent.ConcurrentHashMap<>' with 'ConcurrentHashMap<>' in DefaultFileSystemMaster.
Fixed issue with LraFilterComponent where get() and select() methods may throw a RuntimeException.
Added check for hidden node in NodeInstanceImpl.cancel() method
Added documentation for --bitextrules, --languagemodel, and --word2vecmodel options.
Added check for REPLACEMENT character in StringSerializer to handle 3-byte surrogates / 6-byte surrogate pairs
Fix for handling IllegalStateException in DatastoreImpl
Change getCount() method to use Number instead of Double
Added warning for deprecated project file compilation.
Fixed issue with incorrect lock checking in CmsLocaleGroupService
Removed unused import: jdk.nashorn.api.scripting.ScriptObjectMirror
Fix JRubyAdapter.java to correctly print stack traces
Added a flush() call to VcfImporterService to avoid 'All shards failed' error in Elasticsearch for big datasets
Added environment variable 'cluster.routing.allocation.disk.threshold_enabled' with value 'false' to Elasticsearch container.
Added null checks for close method in ByteSelectiveStreamReader
Added logging for middleware name in App.java
Removed unused import org.unbescape.java.PropertiesEscapeLevel
Added waitAllForSafeState() call after creating new Hazelcast instance
Fixed nonce generation for BTCE API keys by using time-based nonces and increasing them only when necessary.
Update ParserTransformer to use ASMSettings.FRAMES and load extended classes using the parent class's class loader.
Added shutdown logic for PresentsServer
Fixed incorrect logic in LockJDBCConnection.close() method.
Fixed bug in SurfaceLayerGL.java by removing unnecessary null check and cleaning up code.
FramerManager: removed unnecessary Date and SimpleDateFormat usage.
Added null check for getAllHeaders() in RestClient
Update ECKey.java to correctly handle reading DLSequence from ASN.1 stream.
Added role-based access control to HttpSecurityPreparer
Added support for properties in Interpreter.java
Added a null check for FileSystem before creating a new one.
Added preExecute method call to PluginManagerInterceptor to allow for modification of mojo configuration before execution.
Fixed bug in VerifierServletTransaction.java where document was being passed as null

Added test for getInt1() and setInt1() methods in ExampleBean
Update BranchTemplateServiceImpl to use a different replacement function for template instances.
Added null check for output before closing writer in Delombok.java
Added try-catch block to handle exceptions in @ClassRule method
Update PlaceRegistry.java to version 1.16 with updated place manager unmapping.
Fixed issue with incorrect calculation of months in Period class.
Update RegisteredLookupExtractionFn to use nullable optimization flag
Update ServerRemoveHandler to use Runtime stage instead of Immediate stage.
Fix SecretSharing class: remove unnecessary check for n >= k and handle varying lengths of share values
Added camel case splitting and phrasing utilities to U class.
Initialized LanguageToolMenus with LTHeadMenu and ContextMenuInterceptor
Added null check for chains in KeyChainGroup constructor
Added support for P2SH addresses in LegacyAddress class.
JUnitTestRunner.java: Changed catch block from JavaLangThrowable to NSException
Added import statement for AuthoritiesConstants in UserResource.java
Fixed bug in InstructionSet.isDeclaration() method
Added method toString() to class BootChargingStationCommand
Added support for composite datastores in JsfHelper
Removed unused import statement and simplified code in HelpFlags.java
Added getToolStrip method to FeatureAttributeWindow class.
Added CrawljaxPluginsUtil.runPreCrawlingPlugins(browser) before crawler.goToInitialURL()
Fix issue with OutputFileMixin.java - Unable to create directories for output file
Added getFirstHeader and getFirstHeaderValue methods to BasicHttpClient.
Replace write() with writePkg() in DirectLogFetcherTest.
Added delayed system exit with configurable delay
Update CollapsingTitleBar to use a null check before setting background
Throw IOException with error code when memory registration fails
Renamed 'activity_type' to 'type' in Activity.java
Added final modifier to CampAST.emit() and replaced StringWriter with PrintWriter for better performance.
Added test case for optimization rejection with replicated table and partitioned table in outer join with order by.
Added sequencenumber field to CommonLogFields interface
Added protocol validation for URL strings in BoxBrowser.
Renamed CanvasTestBase to abstract class and updated imports
Added blocking functionality to NotificationConsumer
Added support for locale-specific column names in DatabaseValidator
Fix typo in DisposableHelper.dispose() method
 Fixes issue #12345: Improperly closed input/output streams in Streams.java. Closes #12345.
Added JPA bootstrap and updated beans configuration in Setup.java
Replace RxJavaObservableExecutionHook with a static final variable
Added shuffling logic for OOB rows in DRF model
Fixed issue with incorrect row count in FindAllQuery
Added loadClass method to EeClassLoader to fix class loading issue.
Added logging statement for adding icon
Added JMX support to LargeState test
Update AndroidSdk.java to use available platforms in SDK
Modified AndroidServlet.java to change the error code from 13 to 10 for StaleElementReferenceException.
Removed unnecessary checkIfObjectIsModel method and replaced it with a call to isClassModel.
Updated ExchangeCookieUtils.java to use millisecond-based expiration times for cookies.
Added constructor to JAXBUtil to prevent instantiation.
Modified BrowserMobProxyTest to count requests based on Configuration.baseUrl

Added @SuppressWarnings("unchecked") annotation to two test methods in ClassContextTest.
Transaction rule builder: Changed authority rule builder to transaction rule builder.
Update configuration to load properties from classpath resources
Added javax.ejb.Local import and updated DiameterStackMultiplexerAS7 interface to use Local instead of Remote
Fixed issue with terminate token in AmazonServiceCatalogService
Added phenotype validator and individual creation retries to Engine constructor.
Added support for multipart/mixed in Update REST API
Added check for 'groupId' property in KafkaClient.onModification() method.
Added override for toString() method in LazyPatternBy class and removed unnecessary synchronized block
Update logo URL in WebAppDatabasePopulator
ClientSession.java: Changed version from responseSequence to version in QueryRequest builder.
Replace Rand.nextLong() with Math.abs(Rand.nextLong()) in SearchResultServlet
Ignored broken Jmol animation test (#100644)
Added import statement for Collections and updated type parameter of ListItems constructor from Collection<Object> to Collection<?>. Also, removed unnecessary cast to Property and added @SuppressWarnings annotation to silence unchecked warning. Fixed FIXME comment.
Added indirect field to MutableBySettingFieldOfField, and updated setIndirectField() method to use it.
Update import statements for Jakarta EE 9 compatibility.
Add HGVS code generation for SNVs only
Update ArchetypeTest to use m2.repo instead of hardcoded path for Maven repository.
Modified PathMatchingSimpleStorageResourcePatternResolver to always add resource to list, regardless of existence check.
Added field m_receivedFirstFragment to track receipt of first fragment task message
Added @Since tag to ModelObjectWithChildren interface
LossDetector: Simplified activateGap method by removing unnecessary delay generator.
Update EntityBuilder to use parseUnsignedLong instead of parseLong for TextChannel ID
Added support for quoted table names in DiffToChangeLog.

Added check for edit mode before initializing RippleView

Added tree processing callback for TSMeta
Update DeploymentBuilder to use RepositoryService instead of RuntimeService
Added test case for none rows other thread exact two queries other
Fixed issue with SqlgStartStepBarrier.java - now properly generates traversers for start step
Fixed constructor parameter check for List widget.
Fix handling of outstanding messages in Puller
Updated FileStoreLogFactory to use updated threshold value for low storage warning
Added click() method call to select radio button instead of setSelected()
Added support for Gradle 6.2 in GradleCompatibilityExtension
Added SPARQL service description namespace and updated managed namespaces
Utils.java: updated return type of getLargeMemoryClass() from int to long
Added support for batched statements in LoadDataChange
Added support for transient modifier in AdapterUtil
Fixed parsing of label parameter in PerfMonCollector
Added natural order of string representation to UsingToStringOrdering
Update CompilerConfiguration to use Set<String> instead of HashSet<String> for disabled global AST transformations
Changed the number of threads in TestMultiThread from 200 to 100.
Added handling of LINK_REMOTE_DETACH event in Engine.java
Disable testCreateBucketedTable due to flaky failure
Added global lock for bitmap decoding to prevent excessive memory thrashing and OOMs.

Added duration attribute to TransitionSet.java
Added null check for index column list
Added logging of ignored events in FiniteStateMachine
Remove @Ignore annotation for XML parser test cases.
SoundManager.java: Fix for drain() issue causing delay in sound playback.

Fixed loop bounds for leap year in TestLocalDate.java
Added constructor for UserProfile with id and updated email field
Fixed deserialization of EvolutionResult by using IO.object.read instead of manual casting.
Added @Ignore annotation to StageTest.java
Removed unused import statement for ParallelTest annotation.
Changed data type of flags field from short to byte in ExclusiveBufferClaim class.

Added import statement for 'java.util.Objects' and updated rendering mode to use Objects.toString() method.
Added synchronization to ShadowLooper's makeThreadLocalLoopers method to ensure thread-safety.
Added file separator and name to toString method in AbstractFile
Added support for appending HTML content to the report widget.
Ensure flag encoder map contains unique keys
Converted separator to forward slash '/' in SiteMojo.java
Cache deserialized value for ClientDelegatingFuture
Added import statements for SelenideElement and Condition, and updated Describe class to use SelenideElement.
Added methods setData and getData to SessionInfo class
Change plan node CPU time to scheduled time in PlanPrinter

Update MySqlWriterCommands to use LIKE instead of SELECT * FROM
Added method toFileSystemSupport interface to convert Path to URI.
Added weak hash map to avoid leaking Slave instances in SlaveReconnectionWork

Update package name from io.jenetics.ext.engine to io.jenetics.ext and remove unused imports
Fix missing resources in ModelTest
Update BCrypt.java to use Bytes.wrap() for constant-time comparison of raw salt and hash
Added support for mapping node types in DocumentGraphMapper.
Replace JCommander with new JCommander and remove unused variable
Replace 'this.proxy' with '((Object) this.proxy)' in ProxyDefaultMethodHandle.java
Added getUnderscoredIsoCode method to I18n class and updated lexer constructor to use qualified class name
Added check for type parameters in method invocation
ClassFileWriter.java: Fix for bug #630111 - reading header of class file
Added logging of sender address in MulticastService join request handling
Added check for non-empty list of DagNodes before returning running jobs counter

Replace integer ID with string ID in LinkedIn profile URL
WorkerOptions: set default value for workers to 1
FormatterMojo: Skipping except block in Java source code
Added method 'upgrade' to lift security restriction on channel.
Added support for printing transaction information for bidirectional and outbound resource adapters.
Added support for getting culprits from multiple builds
Reevaluate AudioDisconnectEvent firing in WebSocketClient
Flipped buffer when reading object in InstallRequest
Added import statement for com.google.common.base.Preconditions and updated accept method to use it.
Modified SourceCode.java to change the range of characters for highlighting
Added check for Windows platform in ChronicleMapBuilder
Added filter for partitions in HiveMetadata
Update LakeBTC exchange specification to use lakebtc.com instead of www.LakeBTC.com
Added commit() method call to DemoApplication constructor to enable HTTPS support.
RefreshListener: Remove warn log and add a refresh method
Suppressed RuntimeException to Exception in ScheduledReporter#report
Fix for incorrect endColumn calculation in JLanguageTool.
Added method isIdElement(String) to determine if an element is a number-based ID, and updated doDetermineRestfulPath(UrlMappingResource, List<String>) to use it.
Modified GoogleCodingConvention.java to change the logic for determining private methods.
Added parameter 'callback' to WebSocketSubscriptionTransport constructor.
Added null check for ViewProperty newInstance method
Updated FileList.java to change 'error' message to 'cancel' message when no input file name is provided.
Added check for null before creating preference adapter
Added debug logging for interrupted sleep in OCommandExecutorScript
Added Jackson annotations for HostIdentifier class.
Added extra escape character checking in CSVWriter.
Update BasePairParameters.java: remove transient modifier from structure field and add missing ring map entries
Added double parsing for time fields in AbstractResultParser.

Added import statement for ArtifactType and added method getDefaultArtifactType() to SvnConnectorConfiguration class.
Replace HashSet with TreeSet in CommandManager#getAllCommandNames
Replace null with UNDEFINED in TypeDescription#findVariable
Added CacheResultUnavailable exception class and deprecated CacheResultUnavilable
Added support for messenger extensions and fallback URL to WebButton
Added parameter 'true' to readItem method in SessionDataManager.java
Changed logger level from debug to trace for capability discovery
Added support for count method in AbstractRepository by using Spring SpecificationsExecutor
Removed unused method clear() in StandardTitanGraph
Added ingress publication for each member endpoint
Added check for null index name in CFMetaData
Disable testCovtypeGBM
Added name field to GeoJsonObject and implemented getter and setter methods.
Added conductor cycle threshold configuration option to MediaDriver.
Added import statement for 'RejectedExecutionException' and updated 'CleanResourcesTask' to handle execution rejection.
Added check for existence of mgmt-users.properties file in domain directory
Removed unnecessary task state tracker registration
Fix compiler bug by adding a work-around for closeable processor class loader.
GitHubProvider: added displayName fallback to user.id.id when displayName is null
Added method readForRecycle to recycle bytes
Added check for comment being null before adding it to multiPart request
Added test cases for FluxCircuitBreakerOperator
Update copyright year in RepackageMojo.java from 2014 to 2015
Added try-catch block for cursor close in GridFS.java
Added @Documented annotation to J2ObjCIncompatible class.
Added constructor for Meta class with no assigned ID
Fixed sigmoid function and changed float to double in ObjFunction.
Update SSL/TLS version and cipher in SslTest
Update German spelling rules: added 'd zwar' to replace 'uns war', and updated 'aschte' to 'usch'.
Renamed variable 'is' to 'in' in method unpackInt(DataInput) for consistency.
Replace velocityEngines with velocityEngine and remove unnecessary code.
PlannerBenchmarkConfig: Changed 'AUTO' threads from using all available processors minus one to minus two to prevent zero.
Added setters for slow and very slow invocation thresholds in ServiceStatistics
Added config logging and moved ops registrar initialization to before client interface creation.
Update fingerprinting algorithm to use PathTools.getPathsOfLengthUpto instead of PathTools.getPathsOfLength.
Added manual provider check and display of manual document
Added support for unzipping files and updated progress bar
Renamed 'allExtensions' to 'extensions' in UserExtensionsTest.
Added support for sending email from DialogRateMe
Added support for multiple entities in HibernateBundle
Added handling of x-amz-acl header in S3ProxyHandler
Removed unnecessary scroll event handling in MapController.

Added support for verifying loops with multiple exit points.
Added version number display for debugging
Added version number to Transaction class
Removed unnecessary System.out.println statement in JemmyDSL.java

Added forward declaration of private local types in ObjectiveCImplementationGenerator.
Added warmUpPartitions method call in setUp method of SmallClusterTest class.
Added 'true' parameter to Scheduling.SINGLETON option
Fixed dependency tracking for upstream projects
Removed unused imports in WarJSFIntegrationTest.java

Logger level changed from info to finest in SelectorOptimizer.java.
Modified TestUtil.java to improve error handling.
Added alt text to DropDownImageListEditor for accessibility
Added node role filtering to Cluster.getDiskUsageStats()
Update AbstractBeanDefinitionBuilder to use resolved types instead of type arguments
Added --provision flag to VagrantState.RUNNING
Added import statement for org.jboss.as.controller.notification.Notification and added a new method emit(Notification notification) that throws a ReadOnlyContextException.
Added check for record fields with matching names in AvroUtils#equals
Modified AbstractRemoteCommandObject.java to improve error handling in REST invocations.
Added check for empty brand file before adding it to the book.

Remove unused field and methods related to usage count tracking.
Added toString() method to FluxMaterialize class.

Added logging of exception when unable to create AgiScript instance
Update author name from 'lefty' to 'lefteris banos' in ProductRegisterEndPoint.java

Added test for large date format in PactDslJsonBodyTest.
Removed unnecessary print statement in CollapsePropertiesTest.
Update Environment.java to use Msc.isStandalone() instead of Env.hasInitial("mode", "production") for determining production mode.
Update ProfileType.java to reflect changed behavior in QDM regarding profile types and base types.
Added support for appSecurity-5.0 in GrpcServerComponent
Added IBM_UTC_ID to TimeZones class and updated isUtc method to include IBM UTC ID.
Update NUM_FILES_PER_CHUNK from 1000 to 200
Added support for forced glossary and monolingual corpus in LanguageTranslation API
Update NotifyTest.java to support protocol version 2

Added logging statement for Feature Requirement checking in AbstractGremlinTest
Removed unnecessary import statement for 'androidx.appcompat.app.AppCompatActivity' in MainActivity.java.
Renamed JSON properties in SnapshotRestore.java: JSON_PATH, JSON_NONCE, and JSON_DUPLICATES_PATH
Added method toArtifactQuery class to provide a string representation of the object.
Replace created with edited in DiskLruCache.java
Set core pool size and maximum pool size to threadCount in AsyncScalr.
FxmlLoadResult: updated afterNodeLoaded method to use Consumer<NODE> instead of Consumer<Node>
Added support for parsing Cadd scores from chunk files.
Added method buildAll(Predicate<String> classFilter) to InternalKieBuilder interface
Added code to clear up all attachments for a deployment unit.

Added validation for multiple JSON schema repositories
LazilyLoadedCtor: Fix null return value when static initializer sets property
Election: updated leader election algorithm to use a new candidate term ID and persists it to cluster mark file.
Added option to disable summary creation in AnalysisRun.
Remove unused field 'userId' from OrganizationDto.java

Fixed SwissKnife.java: changed toDate method to use LocalDateTime instead of Date
Removed try-catch block in Script.execute(String) and simplified return statement.
Added check for private methods in PartiallyConstructedObjectAccess to prevent false positives.
Update copyright year in AnalyzerPhase.java
Fix corrupted hint name detection in HintedHandOffManager
Fixed progress bar display in IFFImageWriter by deferring last 10% of buffering until image is fully written.
Added path and description to DefaultResourcePersister
Increase open files limit from 8000 to 1024
Added backpressure handling to ChannelDuplexHandler
Remove unnecessary long type check in ShardingProperties
Replace parameter type from Fragment to Context in AndPermission.java
Added assertion for non-empty acquired locks in PidRuntimeFactoryTest
Remove unnecessary calls to edit log in MasterInfo.java

PBDSegment.java: Increased buffer size for truncated entries
Remove unnecessary '!' in SqlStageExecution.java
Removed unnecessary System.out.println statement in ThriftClusterImpl.java
Added check for operator stats being done before closing exchange client
Added boolean return type to TreeNodeLongClickListener interface.
Added check for non-empty list of problems in ConfigException.ValidationFailed
Added temporary configuration file for integration test
Removed unused import statement for 'com.google.common.collect.Streams' and updated code to use 'java.util.stream.StreamSupport' instead.
Added PriorityResource to SingularityResourceModule
Fixed incorrect comparison in FunctionContext.java
Fix for incorrect marking of tokens in XMLRuleHandler
Increased slice size from 4 to 20 in OneShotBenchmark.java
Added support for DB2 databases in AddPrimaryKeyGenerator.
Added nextNextIndex variable and updated logic for handling term IDs in Publication class.
Added ability to specify static constructor name for @Data annotated classes
Added check for joining mode before taking snapshot
Added conditional URLClassLoader for JspC in CompilationMojoSupport
Added debug logging for BitcoinSerializer
Added compareAndSet method to Sequence class and added padding to prevent false sharing
Fixed assertion in ModelTest#testInsert method.
Changed minimum refresh frequency from 3 to 2 for PeriodicalRefreshStrategyTest.
,AsyncExecute.java
Update SearchUI.java to add user information and instance fonts
Added definite assignment for receiver in method type
Update KieControllerSwaggerIntegrationTest to check for alternative title in controller documentation
Update SafeModeManager to use nanoseconds for worker connect wait time
Added tests for encoding with special chars and text
Modified SseBuilder class to validate timeslot duration and alm domain
Increased timeout for tests: testTieBreaker, testCloseServer, and testSendFile
Fixed issue with empty access token in MPRestClient

Fixed a bug in the Task class where the isAvailable() method was incorrectly returning true when all consumable NotifyingPartitionWriters were not available.
Added waitForClusterToForm method call in MultipleCacheManagersTest
Update BeanSpec to use rawTypeOf(theType) instead of rawType.getComponentType() when typeParams is empty
Update PerformanceStateContainerTest to use 'not available' instead of 'ops' in testGetPerformanceNumbers
Removed unnecessary transients and extracted shared state for weights and biases.
ResolveExpressionTypesPass: Continue checking remaining fields for args with unknown or error types, and for List<?> with null or unknown element type.
Added method getPolicyRuleURL to PolicyEvent class
Added X-Text-Delivered header to LogResource
Added method onRequestPermissionsResult to NativeScriptActivity to call JS method with same name.
Added native implementation for Thread.interrupted() method.
Added additional logic to handle PORT_CONFLICT errors in RollingUpdateOpFactory
Updated handleMaxTextLengthRequest to use config.getMaxTextLengthAnonymous() instead of UserLimits.getDefaultLimits(config).getMaxTextLength()
Added deprecation warnings for putAll and keySet methods in ContentMap class.
Fixed bug in RdfUtils.normalize() method by removing unnecessary check for newline or tab characters.
Fixed issue with whitespace handling in XWikiXHTMLWhitespaceXMLFilter
